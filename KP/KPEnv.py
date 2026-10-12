
from dataclasses import dataclass
import torch

from KProblemDef import get_random_problems
import os
import random
import numpy as np
from typing import Union
from ml4co_kit import *
from utils.kp_data import KPSolver


@dataclass
class Reset_State:
    weights: torch.Tensor = None
    # shape: (batch, problem)
    values: torch.Tensor = None
    # shape: (batch, problem)
    pseudo_points: torch.Tensor = None
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    capacity: torch.Tensor = None
    current_item: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, item)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    ref_items: Union[None, torch.Tensor] = None


class KPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.item_cnt = env_params["item_cnt"]
        self.pomo_size = env_params["pomo_size"]

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.weights: torch.Tensor = None
        # shape: (batch, problem)
        self.values: torch.Tensor = None
        # shape: (batch, problem)
        self.pseudo_points: torch.Tensor = None
        # shape: (batch, problem, 2)
        self.task = env_params.get("test_type") or env_params.get("problem_pool")[0]

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_item = None
        # shape: (batch, pomo)
        self.selected_item_list = None # [sequential index format]
        # shape: (batch, pomo, 0~problem)
        self.selected_items = None # [binary decision variable format]
         # shape: (batch, pomo, items)

        # Dynamic-2
        ####################################
        self.capacity = None
        # shape: (batch, pomo)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # STEP-State
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
        self.ref_items = None

        # load data if needed
        self.use_label = env_params.get("use_label")
        if self.use_label:
            self.label_prob = env_params["label_prob"]
            self.train_folder = env_params["train_folder"]
            self.train_sub_files = [
                os.path.join(self.train_folder, train_files) \
                    for train_files in sorted(os.listdir(self.train_folder)) if train_files.endswith(".txt")
            ]
            self.train_sub_files_num = len(self.train_sub_files)
            self.train_data_historty_cache = dict()
            self.train_data_cache = None
            self.train_data_cache_idx = 0
            if self.task == "KP":
                self.tmp_solver = KPSolver()
            else:
                raise NotImplementedError

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.use_label and random.uniform(0, 1) < self.label_prob:
            weights, values, capacities, pseudo_points, self.ref_items = self.generate_kp_data_with_label(batch_size)
        else:
            weights, values, capacities, pseudo_points = get_random_problems(batch_size, self.item_cnt)
            self.ref_items = None


        # Instance Augmentation (validation only)
        #         raise NotImplementedError()

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.weights = weights
        self.reset_state.values = values
        self.reset_state.pseudo_points = pseudo_points
        self.weights = weights
        self.values = values
        self.capacity = capacities[:, None].repeat(1, self.pomo_size)
        self.pseudo_points = pseudo_points

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def load_problems_manual(
            self, weights: torch.Tensor, values: torch.Tensor,
            pseudo_points: torch.Tensor, capacities: torch.Tensor,
            ref_items: torch.Tensor
        ):
        self.batch_size = weights.size(0)
        self.weights = weights
        self.values = values
        self.ref_items = ref_items
        self.pseudo_points = pseudo_points

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.weights = weights
        self.reset_state.values = values
        self.reset_state.pseudo_points = pseudo_points
        self.capacity = capacities[:, None].repeat(1, self.pomo_size)

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX


    def reset(self):
        self.selected_count = 0
        self.current_item = None
        # shape: (batch, pomo)
        self.selected_item_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        # shape: (batch, pomo)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.item_cnt + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.capacity = self.capacity
        self.step_state.current_item = self.current_item
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.ref_items = self.ref_items

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)
        self.selected_count += 1

        self.current_item = selected
        # shape: (batch, pomo)
        self.selected_item_list = torch.cat((self.selected_item_list, self.current_item[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~item)

        # UPDATE STEP STATE
        self.step_state.current_item = self.current_item
        weight_list = self.weights[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, item)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_weight = weight_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.capacity -= selected_weight
        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf") # mask selected items
        # shape: (batch, pomo, item)
        weight_too_large = self.capacity[:, :, None] < weight_list
        # shape: (batch, pomo, item)


        self.ninf_mask[weight_too_large] = float("-inf") # mask overweighted items
        # shape: (batch, pomo, item)

        self.finished = (self.ninf_mask == float("-inf")).all(dim=2)
        # shape: (batch, pomo)

        # do not mask the dummy item for finished episode
        self.ninf_mask[:, :, -1][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.capacity = self.capacity
        self.step_state.current_item = self.current_item
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = self._get_total_value()  # We MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            reward = None

        return self.step_state, reward, done

    def _get_total_value(self):
        selected_idx = self.selected_item_list
        selected_items_dummy = torch.zeros(self.batch_size, self.pomo_size, self.item_cnt + 1)
        # Boolean vector for solution, aligning with ML4CO-Kit format

        selected_items_dummy[
            torch.arange(self.batch_size)[:, None, None].repeat(1, self.pomo_size, selected_idx.shape[-1]).flatten(),
            torch.arange(self.pomo_size)[None, :, None].repeat(self.batch_size, 1, selected_idx.shape[-1]).flatten(),
            selected_idx.flatten()
        ] = 1
        self.selected_items = selected_items_dummy[:, :, :-1]
        total_value = (self.selected_items * self.values[:, None, :-1].repeat(1, self.pomo_size, 1)).sum(2) # shape: [batch, pomo]

        return total_value

    def generate_kp_data_with_label(self, batch_size: int):
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
                print(f"\nload kp train data from {sel_train_sub_file_path}")
                self.tmp_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "weights": self.tmp_solver.weights,
                    "values": self.tmp_solver.values,
                    "capacities": self.tmp_solver.capacities,
                    "ref_items": self.tmp_solver.ref_items_label,
                    "data_size": self.tmp_solver.weights.shape[0]
                }
                self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache

            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size

        # retrieve a portion of data from the cache
        weights = self.train_data_cache["weights"][begin_idx:end_idx]
        values = self.train_data_cache["values"][begin_idx:end_idx]
        capacities = self.train_data_cache["capacities"][begin_idx:end_idx]
        ref_items = self.train_data_cache["ref_items"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx

        # data process
        weights = torch.cat([torch.tensor(weights), torch.zeros(batch_size, 1)], dim=1)  # shape: (batch, item + 1) (with a dummy item)
        values = torch.cat([torch.tensor(values), torch.zeros(batch_size, 1)], dim=1)  # shape: (batch, item + 1) (with a dummy item)
        pseudo_points = torch.stack([weights, values], dim=-1) # shape: (batch, item + 1, 2)
        capacities = torch.tensor(capacities)  # shape: (batch,)
        ref_items = torch.tensor(ref_items)  # shape: (batch, item)

        return weights, values, capacities, pseudo_points, ref_items
