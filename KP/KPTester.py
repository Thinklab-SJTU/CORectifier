
import torch
from torch import Tensor
import numpy as np
from logging import getLogger
from copy import deepcopy

from KPEnv import KPEnv as Env
from KPModel import KPModel as Model

from utils.utils import *
from KProblemDef import *
from tqdm import tqdm, trange
from ml4co_kit import *
from utils.kp_data import KPSolver


class KPTester:
    def __init__(
        self,
        env_params,
        model_params,
        tester_params
    ):

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
        self.tmp_solver = KPSolver()
        self.tmp_solver.from_txt(file_path=tester_params["test_file"], ref=True) #, show_time=True)

        self.num_instances = min(tester_params.get("num_instances", self.tmp_solver.ref_items_label.shape[0]),
                                    self.tmp_solver.ref_items_label.shape[0])
        if self.num_instances < self.tmp_solver.ref_items_label.shape[0]:
            self.tmp_solver.weights = self.tmp_solver.weights[:self.num_instances]
            self.tmp_solver.values = self.tmp_solver.values[:self.num_instances]
            self.tmp_solver.capacities = self.tmp_solver.capacities[:self.num_instances]
            self.tmp_solver.ref_items_label = self.tmp_solver.ref_items_label[:self.num_instances, :]

    def run(self):
        np_items_label = []
        ori_objs = []
        batch_size = 1
        for idx in trange(self.num_instances):
            weights = torch.tensor(self.tmp_solver.weights[idx])  # shape: (item,)
            values = torch.tensor(self.tmp_solver.values[idx])  # shape: (item,)
            weights = torch.cat([weights, torch.zeros(batch_size)], dim=0) # shape: (item+1,)
            values = torch.cat([values, torch.zeros(batch_size)], dim=0) # shape: (item+1,)
            capacities = torch.tensor(self.tmp_solver.capacities[idx])
            obj, selected_idx = self._test_one_instance(weights, values, capacities)
            selected_items_label_dummy = torch.zeros(self.env.item_cnt + 1)
            selected_items_label_dummy[selected_idx] = 1
            selected_items_label = selected_items_label_dummy[:-1].cpu().numpy()
            ori_objs.append(obj)
            np_items_label.append(selected_items_label)
        print("[Test Result] Original Obj:", sum(ori_objs) / len(ori_objs))
        for idx, sel_items in enumerate(np_items_label):
            np_items_label[idx] = np.expand_dims(sel_items, axis=0)
        np_items_label = np.concatenate(np_items_label, axis=0)
        self.tmp_solver.items_label = np_items_label
        metrics = self.tmp_solver.evaluate(calculate_gap=True)
        print("[Test Result]", metrics)

        return metrics


    @torch.no_grad()
    def _test_one_instance(
        self, weights: Tensor, values: Tensor, capacities: Tensor
    ):
        # Augmentation
        ###############################################
        aug_factor = self.tester_params["aug_factor"]
        batch_size = 1

        weights_batched, values_batched, capacities_batched, index_map = augment_items(
            weights.unsqueeze(0), values.unsqueeze(0), capacities.unsqueeze(0), aug_factor
        )
        pseudo_points_batched = torch.stack([weights_batched, values_batched], dim=-1)

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(
                weights_batched,
                values_batched,
                pseudo_points_batched,
                capacities_batched,
                ref_items=None
            ) # set ref_items
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()

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

        max_pomo_reward, best_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo, shape: (augmentation, batch)

        best_pomo_tours = self.env.selected_item_list[torch.arange(aug_factor), best_pomo_idx.view(-1), :]

        no_aug_score = max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, best_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation, shape: (batch,)

        best_aug_tour = best_pomo_tours[best_aug_idx.view(-1), :]

        aug_score = max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        best_tour = index_map[best_aug_idx.item(), best_aug_tour.reshape(-1)]
        best_obj = aug_score.item()

        return best_obj, best_tour
