
from dataclasses import dataclass
import torch
from torch import Tensor

from CVRProblemDef import get_random_problems, augment_data_hybrid

import os
import random
import numpy as np
from typing import Union
from ml4co_kit import *


@dataclass
class Reset_State:
    depots: torch.Tensor = None
    # shape: (batch, 1, 2)
    points: torch.Tensor = None
    # shape: (batch, problem, 2)
    demands: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    capacity: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    ref_tours: Union[None, torch.Tensor] = None


class CVRPEnv:
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
        self.depot_nodes = None
        # shape: (batch, problem+1, 2)
        self.depot_demands = None
        # shape: (batch, problem+1)
        self.task = env_params.get("test_type") or env_params.get("problem_pool")[0]

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.capacity = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
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
            if self.task == "CVRP":
                self.tmp_solver = CVRPSolver()
            else:
                raise NotImplementedError

    # def use_saved_problems(self, filename, device):

    #     loaded_dict = torch.load(filename, map_location=device)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.use_label and random.uniform(0, 1) < self.label_prob:
            depots, points, demands, self.ref_tours = self.generate_cvrp_data_with_label(batch_size)
        else:
            depots, points, demands = get_random_problems(batch_size, self.node_cnt)
            self.ref_tours = None
        if aug_factor > 1:
            if aug_factor in [2, 4, 8] or aug_factor > 8:
                self.batch_size = self.batch_size * aug_factor
                depots = augment_data_hybrid(depots, aug_factor)
                points = augment_data_hybrid(points, aug_factor)
                demands = demands.repeat(aug_factor, 1)
                if self.ref_tours is not None:
                    self.ref_tours = self.ref_tours.repeat(aug_factor, 1)
            else:
                raise NotImplementedError()

        self.depot_nodes = torch.cat((depots, points), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_demands = torch.cat((depot_demand, demands), dim=1)
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depots = depots
        self.reset_state.points = points
        self.reset_state.demands = demands

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def load_problems_manual(self, depots: Tensor, points: Tensor, demands: Tensor, ref_tours: Tensor):
        self.batch_size = depots.shape[0]
        self.depot_nodes = torch.cat((depots, points), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_demands = torch.cat((depot_demand, demands), dim=1)
        # shape: (batch, problem+1)
        self.ref_tours = ref_tours

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depots = depots
        self.reset_state.points = points
        self.reset_state.demands = demands

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.capacity = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_cnt+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_cnt+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.capacity = self.capacity
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.ref_tours = self.ref_tours

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_demands[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.capacity -= selected_demand
        self.capacity[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf")
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.capacity[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float("-inf")
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float("-inf")).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.capacity = self.capacity
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_nodes[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


    def generate_cvrp_data_with_label(self, batch_size: int, flip_ref_tours: bool = True):
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
                print(f"\nload cvrp train data from {sel_train_sub_file_path}")
                self.tmp_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                demands = self.tmp_solver.demands
                capacities = self.tmp_solver.capacities
                demands = demands / capacities.reshape(-1, 1) # normalize for capacity = 1
                self.train_data_cache = {
                    "depots": self.tmp_solver.depots,
                    "points": self.tmp_solver.points,
                    "demands": demands,
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
        demands = self.train_data_cache["demands"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx

        # data process
        depots = torch.tensor(depots, dtype=torch.float32).unsqueeze(1) # shape: (batch, 1, 2)
        points = torch.tensor(points, dtype=torch.float32)  # shape: (batch, node, 2)
        demands = torch.tensor(demands, dtype=torch.float32) # shape: (batch, nodes)
        ref_tours = torch.tensor(ref_tours, dtype=torch.int32)  # shape: (batch, max_tour_length)

        if flip_ref_tours:
            flip_mask = torch.rand(batch_size, device=ref_tours.device) < 0.5
            flip_indices = torch.where(flip_mask)[0]
            valid_len = (ref_tours != -1).sum(-1)
            for idx in flip_indices:
                ref_tours[idx, :valid_len[idx]] = ref_tours[idx][:valid_len[idx]].flip(0)
        return depots, points, demands, ref_tours
