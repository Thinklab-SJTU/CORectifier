import os
import random

from dataclasses import dataclass
import torch
import numpy as np
from utils.ATSProblemDef import get_random_problems
from utils.base_methods import *
from utils.generate_data import gen_3sat, gen_Euclidean, gen_hcp
from utils.positional_encoding import make_positional_encoding_cosh_recur

from ml4co_kit import *
from typing import Union
from copy import deepcopy

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, node, node)


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


class ATSPEnv:
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
        # shape: (batch, pomo, 0~)

        # STEP-State
        ####################################
        self.step_state = None

        # multiple scales
        self.min_scale = env_params.get("min_scale")
        self.max_scale = env_params.get("max_scale")

        self.pos_emb_dim = self.env_params["pos_embedding_dim"]

        self.problem_type_idx = 0

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
            if self.task == "ATSP":
                self.tmp_solver = ATSPSolver()
            elif self.task == "TSP":
                self.tmp_solver = TSPSolver()

    def load_problems_from_pool(self, batch_size):
        self.batch_size = batch_size
        n_problem = len(self.env_params["problem_pool"])
        self.problem_type_idx = np.random.randint(n_problem)
        problem_type = self.env_params["problem_pool"][self.problem_type_idx]
        ref_tours = None
        if problem_type == "ATSP":
            if self.use_label and random.uniform(0, 1) < self.label_prob:
                problems, ref_tours = self.generate_atsp_data_with_label(batch_size)
            else:
                self.node_cnt = np.random.randint(
                    self.env_params["min_scale"],
                    self.env_params["max_scale"])
                problem_gen_params = self.env_params["problem_gen_params"]
                problems = get_random_problems(batch_size, self.node_cnt).cpu().numpy()
        elif problem_type == "TSP":
            if self.use_label and random.uniform(0, 1) < self.label_prob:
                problems, ref_tours = self.generate_tsp_data_with_label(batch_size)
            else:
                self.node_cnt = np.random.randint(
                    self.env_params["min_scale"],
                    self.env_params["max_scale"])
                problems = [gen_Euclidean(self.node_cnt, 2) for _ in range(self.batch_size)]
        elif problem_type == "hcp":
            self.node_cnt = np.random.randint(
                self.env_params["min_scale"],
                self.env_params["max_scale"])
            problems = [gen_hcp(self.node_cnt, np.random.rand() * 0.2 + 0.1) for _ in range(self.batch_size)]
        elif problem_type == "3sat":
            self.node_cnt = self.env_params["node_cnt"]
            if self.node_cnt == 20:
                num_clauses = [3, 2, 2]
                num_vars = [3, 4, 5]
            elif self.node_cnt == 50:
                num_clauses = [3, 4, 5, 6, 7]
                num_vars = [6, 5, 5, 4, 3]
            elif self.node_cnt == 100:
                num_clauses = [9, 8, 7, 6, 5]
                num_vars = [5, 6, 7, 8, 9]
            rand_idx = np.random.randint(0, len(num_clauses))
            self.node_cnt = 2 * num_clauses[rand_idx] * num_vars[rand_idx] + num_clauses[rand_idx]
            problems = [gen_3sat(num_clauses[rand_idx], num_vars[rand_idx]) for _ in range(self.batch_size)]
        else:
            raise NotImplementedError("Problem type {} is not implemented.".format(problem_type))
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        # positional encoding
        self.pos_emb = make_positional_encoding_cosh_recur(self.pos_emb_dim, self.node_cnt, scaler=100)
        self.pos_emb = self.pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # initial tours
        solver = BaseSolver(self.node_cnt)
        if self.env_params["init_solver"] == None:
            pass
        elif self.env_params["init_solver"] == "nn":
            f_solver = solver.solve_nearest_neighbor
        elif self.env_params["init_solver"] == "lkh":
            f_solver = solver.solve_lkh
        elif self.env_params["init_solver"] == "rand":
            f_solver = solver.solve_rand_perm
        elif self.env_params["init_solver"] == "fi":
            f_solver = solver.solve_farthest_insertion
        elif self.env_params["init_solver"] == "ri":
            f_solver = solver.solve_random_insertion
        else:
            raise NotImplementedError("Solver {} is not implemented.".format(self.env_params["init_solver"]))
        if self.env_params["init_solver"]:
            self.ori_problems = deepcopy(problems)
            for i in range(self.batch_size):
                f_solver(problems[i])
                init_tour = np.array(solver.get_path(), dtype=np.int32)
                problems[i] = problems[i][init_tour, :][:, init_tour]
                if ref_tours is not None:
                    ref_tours[i] = torch.tensor([torch.argwhere(torch.tensor(init_tour) == v).item() for v in ref_tours[i]])
        # problems
        self.problems = torch.Tensor(np.array(problems)) if ref_tours is None else problems
        self.ref_tours = ref_tours

    def load_problems_manual(self, problems, solved_tours=None):
        # problems.shape: (batch, node, node)

        self.batch_size = problems.size(0)
        self.node_cnt = problems.size(1)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.pos_emb = make_positional_encoding_cosh_recur(self.pos_emb_dim, self.node_cnt, scaler=100)
        self.pos_emb = self.pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # initial tours
        solver = BaseSolver(self.node_cnt)
        if self.env_params["init_solver"] == None:
            pass
        elif self.env_params["init_solver"] == "nn":
            f_solver = solver.solve_nearest_neighbor
        elif self.env_params["init_solver"] == "lkh":
            f_solver = solver.solve_lkh
        elif self.env_params["init_solver"] == "fi":
            f_solver = solver.solve_farthest_insertion
        elif self.env_params["init_solver"] == "ri":
            f_solver = solver.solve_random_insertion
        else:
            raise NotImplementedError("Solver {} is not implemented.".format(self.env_params["init_solver"]))
        if self.env_params["init_solver"]:
            reperms = []
            self.ori_problems = deepcopy(problems)
            for i in range(self.batch_size):
                f_solver(problems[i])
                init_tour = np.array(solver.get_path(), dtype=np.int64)
                problems[i] = problems[i][init_tour, :][:, init_tour]
                reperms.append(init_tour)
                if solved_tours is not None:
                    solved_tours[i] = torch.tensor([torch.argwhere(torch.tensor(init_tour) == v).item() for v in solved_tours[i]])
        # problems
        self.problems = problems
        self.ref_tours = solved_tours
        # shape: (batch, node, node)
        if self.env_params["init_solver"] is not None:
            self.reperms = reperms
            return reperms

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self._create_step_state()

        reward = None
        done = False
        return Reset_State(problems=self.problems), reward, done

    def _create_step_state(self):
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX, ref_tours=self.ref_tours)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.node_cnt))
        # shape: (batch, pomo, node)

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, node_idx):
        # node_idx.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = node_idx
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~node)

        self._update_step_state()

        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            reward = -self._get_total_distance()  # Note the MINUS Sign ==> We MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            reward = None
        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float("-inf")
        # shape: (batch, pomo, node)

    def _get_total_distance(self):
        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.node_cnt)
        # shape: (batch, pomo, node)
        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance


    def generate_atsp_data_with_label(self, batch_size: int):
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
                    "dists": self.tmp_solver.dists,
                    "ref_tours": self.tmp_solver.ref_tours,
                    "data_size": self.tmp_solver.dists.shape[0]
                }
                self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache

            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size

        # retrieve a portion of data from the cache
        dists = self.train_data_cache["dists"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx

        # data process
        dists = torch.Tensor(dists)
        ref_tours = torch.Tensor(ref_tours[:, :-1])

        return dists, ref_tours

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
        dists = points_to_distmat(torch.tensor(points))
        ref_tours = torch.Tensor(ref_tours[:, :-1])

        return dists, ref_tours
