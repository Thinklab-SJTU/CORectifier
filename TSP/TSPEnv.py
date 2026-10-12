
from dataclasses import dataclass
import torch

from TSProblemDef import get_random_problems, augment_data_8_fold

import os
import random
import numpy as np
from typing import Union
from ml4co_kit import *

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    ref_tours: Union[None, torch.Tensor] = None


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.node_cnt = env_params["node_cnt"]
        self.pomo_size = env_params["pomo_size"]

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)
        self.task = env_params.get("test_type") or env_params.get("problem_pool")[0]

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # STEP-State
        ####################################
        self.step_state = None
        self.ref_tours = None

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
            if self.task == "TSP":
                self.tmp_solver = TSPSolver()
            else:
                raise NotImplementedError

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.use_label and random.uniform(0, 1) < self.label_prob:
            self.problems, self.ref_tours = self.generate_tsp_data_with_label(batch_size)
            # problems.shape: (batch, node, 2), ref_tours.shape: (batch, node)
        else:
            self.problems = get_random_problems(batch_size, self.node_cnt)
            self.ref_tours = None

        # Instance Augmentation (validation only)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_data_8_fold(self.problems)  # shape: (8*batch, node, 2)
                self.ref_tours = self.ref_tours.repeat(8, 1) if self.ref_tours is not None else None  # shape: (8*batch, node)
            else:
                raise NotImplementedError()

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_problems_manual(self, problems, ref_tours):
        self.batch_size = problems.size(0)
        self.problems = problems
        self.ref_tours = ref_tours

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX, ref_tours=self.ref_tours)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.node_cnt))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(problems=self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~node)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float("-inf")
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            reward = -self._get_travel_distance()  # Note the MINUS Sign ==> We MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.node_cnt, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.node_cnt, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


    def generate_tsp_data_with_label(self, batch_size: int):
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
                    "points": self.tmp_solver.points,
                    "ref_tours": self.tmp_solver.ref_tours,
                    "data_size": self.tmp_solver.points.shape[0]
                }
                self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache

            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size

        # retrieve a portion of data from the cache
        points = self.train_data_cache["points"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx

        # data process
        points = torch.tensor(points)  # shape: (batch, node, 2)
        ref_tours = torch.tensor(ref_tours[:, :-1])  # shape: (batch, node)

        return points, ref_tours
