import torch
from torch import Tensor
import numpy as np
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPTester import CVRPTester
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import *

from utils.utils import *

from ml4co_kit import *
from utils.local_search import cvrp_classic_local_search
import math
import random

scheduler_dict = {
    "multi-step": MultiStepLR,
    "cosine": CosineAnnealingLR
}

aug_factor_dict = { # for IL only
    50: 50,
    100: 32,
    200: 16,
    500: 16
}

class CVRPTrainer:
    def __init__(
        self,
        env_params,
        model_params,
        optimizer_params,
        trainer_params
    ):
        # rectify
        self.use_label = env_params["use_label"]
        self.instance_prob = env_params["instance_prob"]
        self.act_prob = env_params["act_prob"]
        self.il_loss = trainer_params["IL_loss"]

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = self.trainer_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params["optimizer"])
        LRScheduler = scheduler_dict[self.optimizer_params["type"]]
        self.scheduler = LRScheduler(self.optimizer, **self.optimizer_params["scheduler"])

        if self.env.use_label:
            if self.il_loss:
                self.env.label_prob = 1.
                self.instance_prob = 1.
                self.act_prob = 1.
            else:
                self.label_scheduler = CosineAnnealingParamScheduler(self.env.label_prob, eta_min=0, T_max=10, name="label_prob")
                self.inst_scheduler = CosineAnnealingParamScheduler(self.instance_prob, eta_min=0.25, T_max=20, name="instance_prob")
                self.act_scheduler = CosineAnnealingParamScheduler(self.act_prob, eta_min=0, T_max=40, name="act_prob")
        else:
            assert not self.il_loss, "IL can be only enabled when `use_label` is True."

        self.task = "CVRP"

        # Restore
        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_fullname = model_load["path"]
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.result_log.set_raw_data(checkpoint["result_log"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = model_load["epoch"]-1
            self.logger.info("Saved Model Loaded !!")

        # utility
        self.time_estimator = TimeEstimator()

        # validation
        if self.task == "CVRP":
            self.tmp_solver = CVRPSolver()
        self.tmp_solver.from_txt(self.trainer_params["val_file"], ref=True, show_time=True)
        self.val_num = min(trainer_params["val_num"], len(self.tmp_solver.points))
        self.best_val_score = 1e6

    def run(self):
        self._validation()

        self.time_estimator.reset(self.start_epoch)
        val_interval = self.trainer_params["val_interval"]
        for epoch in range(self.start_epoch, self.trainer_params["epochs"]+1):
            self.logger.info("=================================================================")

            # Param scheduler
            self.scheduler.step()
            if self.env.use_label and not self.il_loss:
                self.instance_prob = self.inst_scheduler.step(epoch)
                self.env.label_prob = self.label_scheduler.step(epoch)
                self.act_prob = self.act_scheduler.step(epoch)

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, train_score)
            self.result_log.append("train_loss", epoch, train_loss)

            # Validation
            if epoch % val_interval == 0:
                val_score = self._validation()
                self.result_log.append("val_score", epoch, val_score)
                self.logger.info(f"Epoch {epoch:3d}: Val    Score: {val_score:.4f}")
                if val_score < self.best_val_score:
                    self.best_val_score = val_score
                    self.logger.info("Saving new best_model upon validation")
                    checkpoint_dict = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "result_log": self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, f"{self.result_folder}/checkpoint-{epoch}-{val_score:.4f}.pt")

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params["epochs"], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params["epochs"])
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]

            if epoch % model_save_interval == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict,
                    f"{self.result_folder}/checkpoint-{epoch}.pt")

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, "{}/checkpoint-{}.pt".format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params["train_episodes"]
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params["train_batch_size"], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info("Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}"
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info(f"Epoch {epoch:3d}: Train  Score: {score_AM.avg:.4f},  Loss: {loss_AM.avg:.4f}")
        if self.use_label:
            self.logger.info(f"Label prob: {self.env.label_prob:.4f}, Instance prob: {self.instance_prob:.4f}, Act prob: {self.act_prob}")

        return score_AM.avg, loss_AM.avg

    def _validation(self) -> float:
        tester = CVRPTester.__new__(CVRPTester)
        tester.env = self.env
        tester.model = self.model
        tester.tester_params = {"aug_factor": self.trainer_params["aug_factor"], "augmentation_enable": True}
        self.model.eval()
        objs = []
        for idx in range(self.val_num):
            depot = self.tmp_solver.depots[idx]
            points = self.tmp_solver.points[idx]
            demands = self.tmp_solver.demands[idx] / self.tmp_solver.capacities[idx]
            obj, tour = tester._test_one_instance(
                torch.tensor(depot).unsqueeze(0), torch.tensor(points), torch.tensor(demands)
            )
            if self.trainer_params["use_ls"]:
                tour = cvrp_classic_local_search(init_tour=tour, depot=depot, points=points, demands=demands).reshape(-1)
                coords = np.concatenate((depot[None, :], points), axis=0)
                obj = float(np.linalg.norm(coords[tour[1:]] - coords[tour[:-1]], axis=-1).sum())
            objs.append(obj)
        self.model.train()
        return sum(objs) / len(objs)

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size, aug_factor=aug_factor_dict[self.env.node_cnt] if self.il_loss else 1)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        batch_size, pomo_size = self.env.batch_size, self.env.pomo_size
        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        pomo_ref_tours = None
        if state.ref_tours is not None:
            pomo_ref_tours = _get_rolled_pomo_ref_tours(self.env.ref_tours, batch_size, pomo_size) # (batch * pomo, N)
            state.ref_tours = pomo_ref_tours

        if self.env.use_label and not self.il_loss: # RRL
            rectify_inst_mask = None
            rectify_act_mask = None
            while not done:
                if state.ref_tours is not None:
                    rectify_inst_mask = torch.rand(batch_size * self.env.pomo_size) < self.instance_prob
                    if self.env.current_node is None:  # starting 0
                        rectify_act_mask = torch.zeros_like(rectify_inst_mask).bool() # allow most POMO exploration at the second step
                        pomo_to_select = torch.arange(1, pomo_size + 1)[None, :].expand(batch_size, pomo_size).reshape(-1)
                        next_in_ref = state.ref_tours[:, 1]
                        rectify_act_mask[torch.where(pomo_to_select == next_in_ref)[0]] = True
                    elif self.env.selected_count > 1:
                        act_mask_update = torch.rand(batch_size * pomo_size) < self.act_prob
                        rectify_act_mask = torch.where(
                            self.env.current_node.reshape(-1) == 0,
                            act_mask_update,
                            rectify_act_mask
                        )
                selected, prob = self.model.forward(state, rectify_inst_mask, rectify_act_mask, IL_flag=self.il_loss)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        else: # RL or IL
            while not done:
                selected, prob = self.model.forward(state, IL_flag=self.il_loss)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        # Loss
        ###############################################
        if state.ref_tours is not None and self.il_loss: # pure imitation learning
            advantage = 1.
        else:
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value


        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()


        return score_mean.item(), loss_mean.item()

def _get_rolled_pomo_ref_tours(
    ref_tours: Tensor,
    batch_size: int,
    pomo_size: int
) -> Tensor:
    batch_size, max_tour_len = ref_tours.shape
    pomo_ref_tours = ref_tours.unsqueeze(1).repeat(1, pomo_size, 1).reshape(batch_size * pomo_size, -1)
    rolled_ref_tours = pomo_ref_tours.clone()

    for idx in range(rolled_ref_tours.shape[0]):
        tour = pomo_ref_tours[idx]
        valid_len = (tour != -1).sum() - 1 # the last 0 is the same as the first
        valid_tour = tour[:valid_len].clone()
        depot_positions = torch.where(valid_tour == 0)[0]
        random_depot_idx = depot_positions[torch.randint(len(depot_positions), (1,))].item()
        rolled_valid = torch.roll(valid_tour, shifts=-random_depot_idx, dims=0)
        rolled_ref_tours[idx, :valid_len] = rolled_valid
    return rolled_ref_tours


class CosineAnnealingParamScheduler:
    def __init__(self, base_value, eta_min, T_max, name=None):
        self.base_value = base_value
        self.eta_min = eta_min
        self.T_max = T_max
        self.name = name

    def step(self, epoch):
        return self.eta_min + 0.5 * (self.base_value - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max))
