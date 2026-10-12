
import torch
from torch import Tensor
import numpy as np
from logging import getLogger
from copy import deepcopy

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *
from TSProblemDef import *
from tqdm import tqdm, trange
from ml4co_kit import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = self.tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params["model_load"]
        checkpoint = torch.load(model_load, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # utility
        self.time_estimator = TimeEstimator()
        self.tmp_solver = TSPSolver()
        self.tmp_solver.from_txt(file_path=tester_params["test_file"], ref=True) #, show_time=True)
        self.use_2opt = tester_params["use_2opt"]

        self.num_instances = min(tester_params.get("num_instances", self.tmp_solver.ref_tours.shape[0]),
                                    self.tmp_solver.ref_tours.shape[0])
        if self.num_instances < self.tmp_solver.ref_tours.shape[0]:
            self.tmp_solver.ori_points = self.tmp_solver.points[:self.num_instances, :, :]
            self.tmp_solver.points = self.tmp_solver.points[:self.num_instances, :, :]
            self.tmp_solver.ref_tours = self.tmp_solver.ref_tours[:self.num_instances, :]

    def run(self):
        np_tours = []
        ori_objs = []
        for idx in trange(self.num_instances):
            points_tensor = torch.tensor(self.tmp_solver.points[idx])
            obj, tour = self._test_one_instance(points_tensor) #, self.rect_iters, self.rect_ratio)
            tour = tour.reshape(-1)
            ori_objs.append(obj)
            if self.use_2opt:
                tour = tour.cpu().numpy()
                tour = tsp_mcts_local_search(
                    init_tours=tour, # .cpu().numpy(),
                    heatmap=np.zeros((self.env.node_cnt, self.env.node_cnt)),
                    points=self.tmp_solver.points[idx],
                    time_limit=0.0, # set time limit as 0.0 is equal to 2opt
                    type_2opt=2
                )
            # tour = np.expand_dims(tour, axis=0)
            np_tours.append(tour)
        print("[Test Result] Obj w/o 2-opt:", sum(ori_objs) / len(ori_objs))
        for idx, tour in enumerate(np_tours):
            if isinstance(tour, torch.Tensor):
               tour = tour.cpu().numpy()
            np_tours[idx] = np.expand_dims(tour, axis=0)
        np_tours = np.concatenate(np_tours, axis=0)
        if np_tours.shape[-1] == self.env.node_cnt:
            np_tours = np.concatenate((np_tours, np_tours[:, :1]), axis=1)
        self.tmp_solver.tours = np_tours
        metrics = self.tmp_solver.evaluate(calculate_gap=True)
        print("[Test Result]", metrics)

        return metrics


    @torch.no_grad()
    def _test_one_instance(self, points: Tensor):

        # Augmentation
        ###############################################
        aug_factor = self.tester_params["aug_factor"]
        problems_batched = points.unsqueeze(0)
        if aug_factor == 8:
            problems_batched = augment_data_8_fold(problems_batched)  # shape: (8*batch, node, 2)
        elif aug_factor > 1:
            problems_batched = augment_data_hybrid(problems_batched, aug_factor)

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problems_batched, ref_tours=None) # set ref_tours
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        batch_size = 1

        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, best_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        best_pomo_tours = self.env.selected_node_list[torch.arange(aug_factor), best_pomo_idx.view(-1), :]

        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, best_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)

        best_aug_tour = best_pomo_tours[best_aug_idx.view(-1), :]

        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        best_tour = best_aug_tour # .cpu().numpy()
        best_obj = aug_score.item()

        return best_obj, best_tour
