import torch
import numpy as np
import random
from copy import deepcopy

from logging import getLogger
from torch import Tensor
from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model

from utils.utils import get_result_folder, TimeEstimator

from tqdm import trange
from ml4co_kit import *
from utils.base_methods import *
from scipy.spatial.distance import cdist

class ATSPTester:
    def __init__(
        self,
        env_params: dict,
        model_params: dict,
        tester_params: dict
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
        self.task = env_params["test_type"][0]

        # Restore
        model_load = self.tester_params["model_load"]
        checkpoint = torch.load(model_load, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # utility
        self.time_estimator = TimeEstimator()
        if self.task == "ATSP":
            self.tmp_solver = ATSPSolver()
        elif self.task == "TSP":
            self.tmp_solver = TSPSolver()
        self.tmp_solver.from_txt(file_path=tester_params["test_file"], ref=True, show_time=True)
        self.use_2opt = tester_params["use_2opt"]

        self.num_instances = min(tester_params.get("num_instances", self.tmp_solver.ref_tours.shape[0]),
                                    self.tmp_solver.ref_tours.shape[0])
        if self.num_instances < self.tmp_solver.ref_tours.shape[0]:
            if self.task == "ATSP":
                self.tmp_solver.ori_dists = self.tmp_solver.dists[:self.num_instances, :, :]
                self.tmp_solver.dists = self.tmp_solver.dists[:self.num_instances, :, :]
                self.tmp_solver.ref_tours = self.tmp_solver.ref_tours[:self.num_instances, :]
            elif self.task == "TSP":
                self.tmp_solver.ori_points = self.tmp_solver.points[:self.num_instances, :, :]
                self.tmp_solver.points = self.tmp_solver.points[:self.num_instances, :, :]
                self.tmp_solver.ref_tours = self.tmp_solver.ref_tours[:self.num_instances, :]

        self.rect_iters = tester_params["rectify_iters"]
        self.rect_ratio = tester_params["rectify_ratio"]
        self.rect_lens = tester_params["rectify_lens"]

    def run(self):
        np_tours = []
        ori_objs = []
        for idx in trange(self.num_instances):
            if self.task == "ATSP":
                dist_tensor = torch.tensor(self.tmp_solver.dists[idx])
            elif self.task == "TSP":
                dist_tensor = points_to_distmat(torch.tensor(self.tmp_solver.points[idx]))
            obj, tour, reperm = self._test_one_instance(dist_tensor, self.rect_iters, self.rect_ratio)
            if reperm is not None:
                tour_back = reperm[tour.reshape(-1)]
            else:
                tour_back = tour.reshape(-1)
            ori_objs.append(obj)
            if self.use_2opt:
                if self.task == "ATSP":
                    tour_back = atsp_2opt_local_search(tour_back, self.tmp_solver.dists[idx])
                    # tour_back = eax_2opt_local_search(tour_back, self.tmp_solver.dists[idx])
                elif self.task == "TSP":
                    # points = self.tmp_solver.points[idx]
                    tour_back = tsp_mcts_local_search(
                        init_tours=tour_back,
                        # heatmap=-to_numpy(points_to_distmat(to_tensor(points))),
                        heatmap=np.zeros((self.env.node_cnt, self.env.node_cnt)),
                        points=self.tmp_solver.points[idx],
                        time_limit=0.0, # set time limit as 0.0 is equal to 2opt
                        type_2opt=2
                    )
                    # tour_back = tsp_2opt_local_search(tour_back, self.tmp_solver.points[idx], device=dist_tensor.device)
                    # tour_back = eax_local_search(tour_back, points)

            tour_back = np.expand_dims(tour_back, axis=0)
            if tour_back.shape[-1] == self.env.node_cnt:
                assert sorted(tour_back[0].tolist()) == [i for i in range(self.env.node_cnt)]
            else:
                assert sorted(tour_back[0].tolist()[:-1]) == [i for i in range(self.env.node_cnt)]
            np_tours.append(tour_back)
        print("origianl length: ", sum(ori_objs) / len(ori_objs))
        np_tours = np.concatenate(np_tours, axis=0)
        if np_tours.shape[-1] == self.env.node_cnt:
            np_tours = np.concatenate((np_tours, np_tours[:, :1]), axis=1)
        self.tmp_solver.tours = np_tours
        print(self.tmp_solver.evaluate(calculate_gap=True))


    @torch.no_grad()
    def _test_one_instance(self, dist: Tensor, n_iters: int = 1, rect_prob: float = None):
        dist = dist.unsqueeze(0) # [1, N, N]
        best_tour = None
        best_reperm = None
        best_obj = 1e6

        # Augmentation
        ###############################################
        if self.tester_params["augmentation_enable"]:
            aug_factor = self.tester_params["aug_factor"]
            problems_batched = dist.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            solved_tours = None
            for iter in range(n_iters):
                reperms = self.env.load_problems_manual(deepcopy(problems_batched), solved_tours=solved_tours) # set ref_tours

                #     cost = 0
                #     ref_tours = self.env.ref_tours
                #         cost += self.env.problems[0, ref_tours[0, i], ref_tours[0, i+1]]
                #     cost += self.env.problems[0, ref_tours[0, -1], ref_tours[0, 0]]


                reset_state, _, _ = self.env.reset()
                self.model.pre_forward_pe(reset_state, self.env.pos_emb)


                # Prepare for Rectification
                ###############################################
                if solved_tours is not None:
                    # rect_level = int(self.env.node_cnt * random.uniform(*rect_range))
                    # global_act_mask = torch.zeros(aug_factor * self.env.node_cnt, self.env.node_cnt).bool()
                    # global_act_mask[:, :self.rect_lens] = True

                    global_act_mask = self._create_global_rectify_mask(
                        num_lines=aug_factor * self.env.pomo_size,
                        num_nodes=self.env.node_cnt,
                        type="const",
                        ratio=self.rect_ratio,
                        rect_lens=self.rect_lens
                    )
                    instance_mask = torch.ones(aug_factor * self.env.pomo_size).bool()

                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                while not done:
                    if solved_tours is not None:
                        act_mask = global_act_mask[:, self.env.selected_count].reshape(-1)
                        selected, _ = self.model.forward(
                            state,
                            rectify_mask=instance_mask,
                            act_mask=act_mask
                        )
                    else:
                        selected, _ = self.model.forward(state)
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

                # best_aug_tour = best_aug_tour

                reperm = reperms[best_aug_idx.item()] if reperms is not None else None

                if iter == 0:
                    init_obj = aug_score.item()

                if aug_score < best_obj:
                    best_obj = aug_score.item()
                    best_tour = best_aug_tour.cpu().numpy()
                    best_reperm = reperm
                    if n_iters > 1:
                        solved_tours = torch.tensor(best_reperm)[best_tour.reshape(-1)].expand(aug_factor, -1)


            return best_obj, best_tour, best_reperm

    def repermute_dist(self, dist, perm_type="nn"):
        n = dist.shape[-1]
        if perm_type == "rand":
            reperm = torch.randperm(n)
        elif perm_type == "nn":
            solver = BaseSolver(n, lib_path="./utils/libtsp.so")
            solver.solve_nearest_neighbor(dist)
            reperm = np.array(solver.get_path(), dtype=np.int64)
        return dist[reperm, :][:, reperm], reperm


    def _create_global_rectify_mask(
        self,
        num_lines: int,
        num_nodes: int,
        type: str = "const",
        ratio: float = 0.1,
        rect_lens: Union[int, list] = 5
    ):
        mask = torch.zeros((num_lines, num_nodes), dtype=torch.bool)
        if type == "const":
            assert isinstance(rect_lens, int)
            num_splits = num_nodes // rect_lens
            roll_shifts = [random.randint(1, num_nodes - 1) for _ in range(num_lines)]
            selected_splits = random.sample(range(num_splits), max(int(ratio * num_splits), 1))
            for split_idx in selected_splits:
                mask[:, split_idx * rect_lens: (split_idx + 1) * rect_lens] = True
            for idx in range(num_lines):
                mask[idx] = mask[idx].roll(roll_shifts[idx])
        elif type == "multiple":
            pass

        return mask
