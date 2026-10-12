import torch
from logging import getLogger

from KPEnv import KPEnv as Env
from KPTester import KPTester
from KPModel import KPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import *

from utils.utils import *

from ml4co_kit import *
from utils.kp_data import KPSolver
import math
from tqdm import tqdm, trange

from KProblemDef import *

scheduler_dict = {
    "multi-step": MultiStepLR,
    "cosine": CosineAnnealingLR
}

class KPTrainer:
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
            else:
                self.label_scheduler = CosineAnnealingParamScheduler(self.env.label_prob, eta_min=0, T_max=10, name="label_prob")
                self.inst_scheduler = CosineAnnealingParamScheduler(self.instance_prob, eta_min=0, T_max=20, name="instance_prob")

        self.task = "KP"

        # Restore
        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_fullname = model_load["path"]
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.scheduler.last_epoch = model_load["epoch"] - 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if not model_load["only_weight"]:
                self.result_log.set_raw_data(checkpoint["result_log"])
            self.logger.info("Saved Model Loaded !!")


        # utility
        self.time_estimator = TimeEstimator()

        # validation
        if self.task == "KP":
            self.tmp_solver = KPSolver()
        self.tmp_solver.from_txt(self.trainer_params["val_file"], ref=True, show_time=True)
        self.val_num = min(trainer_params["val_num"], len(self.tmp_solver.weights))
        self.best_val_score = 0

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

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, train_score)
            self.result_log.append("train_loss", epoch, train_loss)

            # Validation
            if epoch % val_interval == 0:
                val_score = self._validation()
                self.result_log.append("val_score", epoch, val_score)
                self.logger.info(f"Epoch {epoch:3d}: Val    Score: {val_score:.4f}")
                if val_score > self.best_val_score:
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
            self.logger.info(f"Label prob: {self.env.label_prob:.4f}, Instance prob: {self.instance_prob:.4f}") #, Rectified items: {self.rect_level}")

        return score_AM.avg, loss_AM.avg

    def _validation(self) -> float:
        tester = KPTester.__new__(KPTester)
        tester.env = self.env
        tester.model = self.model
        tester.tester_params = {"aug_factor": self.trainer_params["aug_factor"], "augmentation_enable": True}
        self.model.eval()
        objs = []
        for idx in range(self.val_num):
            weights = torch.cat((torch.tensor(self.tmp_solver.weights[idx]), torch.zeros(1)))
            values = torch.cat((torch.tensor(self.tmp_solver.values[idx]), torch.zeros(1)))
            obj, _ = tester._test_one_instance(weights, values, torch.tensor(self.tmp_solver.capacities[idx]))
            objs.append(obj)
        self.model.train()
        return sum(objs) / len(objs)

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        batch_size, pomo_size = self.env.batch_size, self.env.pomo_size
        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        rectify_mask_instance = torch.rand(batch_size * self.env.pomo_size) < self.instance_prob if state.ref_items is not None else None
        # shape: (batch * pomo, )
        if not self.use_label:
            while not done:
                selected, prob = self.model.forward(state)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        else:
            if self.il_loss:
                assert rectify_mask_instance.all()
            while not done:
                selected, prob = self.model.forward(state, rectify_mask_instance)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        if state.ref_items is not None and self.il_loss: # pure imitation learning
            assert self.instance_prob == 1.
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
        score_mean = max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()


class CosineAnnealingParamScheduler:
    def __init__(self, base_value, eta_min, T_max, name=None):
        self.base_value = base_value
        self.eta_min = eta_min
        self.T_max = T_max
        self.name = name

    def step(self, epoch):
        return self.eta_min + 0.5 * (self.base_value - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max))
