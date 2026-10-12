#!/usr/bin/env python

import os
import json
import pprint as pp
import numpy as np
import random
import math
from utils.pctsp_data import PCTSPSolver

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        # TensorBoard is optional.
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir, exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    cuda_device_num = getattr(opts, "cuda_device_num", 0)
    opts.device = torch.device(f"cuda:{cuda_device_num}" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1 and not getattr(opts, "single_gpu", False):
        model = torch.nn.DataParallel(model)

    # Initialize baseline
    baseline = NoBaseline()
    if not opts.eval_only:
        if opts.baseline == 'exponential':
            baseline = ExponentialBaseline(opts.exp_beta)
        elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
            assert problem.NAME == 'tsp', "Critic only supported for TSP"
            baseline = CriticBaseline(
                (
                    CriticNetworkLSTM(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.tanh_clipping
                    )
                    if opts.baseline == 'critic_lstm'
                    else
                    CriticNetwork(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.normalization
                    )
                ).to(opts.device)
            )
        elif opts.baseline == 'rollout':
            baseline = RolloutBaseline(model, problem, opts)
        else:
            assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
            baseline = NoBaseline()

    if not opts.eval_only:
        if opts.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    if opts.load_path is not None or opts.resume is not None:
        load_path = opts.load_path if opts.load_path is not None else opts.resume
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

        get_inner_model(model).load_state_dict({**get_inner_model(model).state_dict(), **load_data.get('model', {})})


    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        print("Resuming after epoch {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

        if 'lr_scheduler' in load_data:
            lr_scheduler.load_state_dict(load_data['lr_scheduler'])
            print("  [*] Loaded lr_scheduler state from checkpoint.")
        else:
            print("  [*] Manually advancing lr_scheduler state...")
            for _ in range(epoch_resume + 1):
                lr_scheduler.step()

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

        baseline.epoch_callback(model, epoch_resume)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
        augment_mode=opts.augment_mode,
        augment_num=opts.augment_num
    )

    if opts.il_loss:
        opts.label_prob = 1.0
        opts.instance_prob = 1.0
        opts.act_prob = 1.0
    env = PCTSPEnv(opts)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                env,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


class PCTSPEnv:
    def __init__(self, opts):
        self.graph_size = opts.graph_size
        self.batch_size = opts.batch_size

        self.tmp_solver = PCTSPSolver()

        self.use_label = opts.use_label
        self.il_loss = opts.il_loss
        if self.use_label:
            self.label_prob = opts.label_prob
            self.train_folder = opts.train_folder
            self.train_sub_files = [
                os.path.join(self.train_folder, train_files) \
                    for train_files in sorted(os.listdir(self.train_folder)) if train_files.endswith(".txt")
            ]
            self.train_sub_files_num = len(self.train_sub_files)
            self.train_data_historty_cache = dict()
            self.train_data_cache = None
            self.train_data_cache_idx = 0

            if not self.il_loss:
                self.label_scheduler = CosineAnnealingParamScheduler(opts.label_prob, eta_min=0, T_max=10, name="label_prob")
                self.inst_scheduler = CosineAnnealingParamScheduler(opts.instance_prob, eta_min=0, T_max=20, name="instance_prob")
                self.rect_range_scheduler_low = CosineAnnealingParamScheduler(opts.rect_range[0], eta_min=0, T_max=40, name="range_low")
                self.rect_range_scheduler_high = CosineAnnealingParamScheduler(opts.rect_range[1], eta_min=0.02, T_max=40, name="range_high")

    def load_problem(self, batch_size: int):
        self.batch_size = batch_size
        if self.use_label and random.uniform(0, 1) < self.label_prob:
            self.data = self.generate_pctsp_data_with_label(batch_size)
        else:
            self.data = self.get_random_problems(batch_size, self.graph_size)

    def generate_pctsp_data_with_label(self, batch_size: int):
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]

            # check if the data is in the cache when store_data is True
            if sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else:
                # load data from the train file
                print(f"\nload tsp train data from {sel_train_sub_file_path}")
                self.tmp_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "depots": self.tmp_solver.depots,
                    "points": self.tmp_solver.points,
                    "penalties": self.tmp_solver.penalties,
                    "norm_prizes": self.tmp_solver.norm_prizes,
                    "ref_tours": self.tmp_solver.ref_tours,
                    "data_size": self.tmp_solver.depots.shape[0]
                }
                self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache

            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size

        # retrieve a portion of data from the cache
        depots = self.train_data_cache["depots"][begin_idx:end_idx]
        points = self.train_data_cache["points"][begin_idx:end_idx]
        penalties = self.train_data_cache["penalties"][begin_idx:end_idx]
        norm_prizes = self.train_data_cache["norm_prizes"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx

        ref_tours = [torch.tensor(tour, dtype=torch.long)[:-1] for tour in ref_tours]
        ref_tours = pad_sequence(ref_tours, batch_first=True, padding_value=-1)

        # data process
        depots = torch.Tensor(depots)
        points = torch.Tensor(points)
        penalties = torch.Tensor(penalties)
        norm_prizes = torch.Tensor(norm_prizes)
        ref_tours = torch.Tensor(ref_tours)

        return {
            'depots': depots,
            'points': points,
            'penalties': penalties,
            'norm_prizes': norm_prizes,
            'ref_tours': ref_tours
        }


    def get_random_problems(self, batch_size: int, graph_size: int):
        # depots
        depots = np.random.uniform(size=(batch_size, 2))

        # points
        points = np.random.uniform(size=(batch_size, graph_size, 2))

        # penalties
        penalty_max = 3.0 * 3.0 / graph_size
        penalties = penalty_max * np.random.uniform(size=(batch_size, graph_size))

        # norm_prizes
        prize_max = 4.0 / graph_size
        norm_prizes = prize_max * np.random.uniform(size=(batch_size, graph_size))

        # data process
        depots = torch.Tensor(depots)
        points = torch.Tensor(points)
        penalties = torch.Tensor(penalties)
        norm_prizes = torch.Tensor(norm_prizes)

        return {
            'depots': depots,
            'points': points,
            'penalties': penalties,
            'norm_prizes': norm_prizes
        }


class CosineAnnealingParamScheduler:
    def __init__(self, base_value, eta_min, T_max, name=None):
        self.base_value = base_value
        self.eta_min = eta_min
        self.T_max = T_max
        self.name = name

    def step(self, epoch):
        return self.eta_min + 0.5 * (self.base_value - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run(get_options())
