import os
import time
from tqdm import tqdm
import torch
import math
import random

from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values, log_to_file
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    start_time = time.time()
    cost = rollout(model, dataset, opts)
    if opts.augment_mode is not None:
        augment_num = 8 if opts.augment_mode == '8fold' else opts.augment_num
        cost = cost.reshape(augment_num, -1).min(dim=0).values
    time_duration = time.time() - start_time
    ref_cost = read_ref_cost(opts.val_cost_path)[:len(cost)]
    avg_cost = cost.mean()
    avg_ref_cost = ref_cost.mean()
    gap = (cost - ref_cost) / ref_cost * 100
    avg_gap = gap.mean()
    std_gap = torch.std(gap) / math.sqrt(len(gap))
    log_message = f"[Validation] avg_cost:{avg_cost}, avg_ref_cost:{avg_ref_cost}, gap:{avg_gap}% +- {std_gap}%, time:{time_duration}s"
    print(log_message)
    log_to_file(log_message, opts)

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    if isinstance(dataset, Dataset):
        return torch.cat([
            eval_model_bat(bat)
            for bat
            in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
        ], 0)
    else:
        return eval_model_bat(dataset)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, env, val_dataset, problem, tb_logger, opts):
    print("==============================================================")
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.add_scalar('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = FakeDataset(data_size=opts.epoch_size)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, _ in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        env.load_problem(opts.batch_size)
        batch = baseline.wrap_dataset(env.data)

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            env,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.add_scalar('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    if opts.use_label and not opts.il_loss:
        opts.instance_prob = env.inst_scheduler.step(epoch)
        opts.label_prob = env.label_scheduler.step(epoch)
        opts.rect_range[1] = env.rect_range_scheduler_high.step(epoch)
        opts.rect_range[0] = env.rect_range_scheduler_low.step(epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        env,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    if opts.il_loss:
        rect_level = env.graph_size
    else:
        rect_level = int(env.graph_size * random.uniform(*opts.rect_range))
    global_act_mask = torch.zeros(env.batch_size, env.graph_size).bool().to(opts.device)
    global_act_mask[:, :rect_level] = True
    # shape: (batch, node)

    rectify_mask_instance = torch.rand(env.batch_size) < opts.instance_prob
    # shape: (batch, )
    rectify_mask_instance = rectify_mask_instance.to(opts.device)
    if opts.il_loss:
        assert global_act_mask.all() and rectify_mask_instance.all()

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x, rectify_mask_instance, global_act_mask, opts.act_prob)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    if 'ref_tours' in x.keys() and opts.il_loss:
        advantage = -1.0
    else:
        advantage = cost - bl_val
    reinforce_loss = (advantage * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx: int):
        return torch.tensor([idx])


def read_ref_cost(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    ref_cost = torch.tensor([float(line.strip()) for line in lines])
    return ref_cost
