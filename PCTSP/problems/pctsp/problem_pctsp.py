from torch.utils.data import Dataset
import torch
import os
from problems.pctsp.state_pctsp import StatePCTSP
from utils.beam_search import beam_search

from utils.pctsp_data import PCTSPSolver
import numpy as np
import math


class PCTSP(object):

    NAME = 'pctsp'  # Prize Collecting TSP, without depot, with penalties

    @staticmethod
    def _get_costs(dataset, pi, stochastic=False):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"

        prize = dataset['stochastic_prize'] if stochastic else dataset['norm_prizes']
        prize_with_depot = torch.cat(
            (
                torch.zeros_like(prize[:, :1]),
                prize
            ),
            1
        )
        p = prize_with_depot.gather(1, pi)

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
            (p.sum(-1) >= 1 - 1e-4) |
            (sorted_pi.size(-1) - (sorted_pi == 0).int().sum(-1) == dataset['points'].size(-2))
        ).all(), "Total prize does not satisfy min total prize"
        penalty_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['penalties'][:, :1]),
                dataset['penalties']
            ),
            1
        )
        pen = penalty_with_depot.gather(1, pi)

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depots'][:, None, :], dataset['points']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        length = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
            + (d[:, 0] - dataset['depots']).norm(p=2, dim=-1)  # Depot to first
            + (d[:, -1] - dataset['depots']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
        )
        # We want to maximize total prize but code minimizes so return negative
        # Incurred penalty cost is total penalty cost - saved penalty costs of nodes visited
        return length + dataset['penalties'].sum(-1) - pen.sum(-1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PCTSPDataset(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        # With beam search we always consider the deterministic case
        state = PCTSPDet.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class PCTSPDet(PCTSP):

    @staticmethod
    def get_costs(dataset, pi):
        return PCTSP._get_costs(dataset, pi, stochastic=False)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCTSP.initialize(*args, **kwargs, stochastic=False)


class PCTSPStoch(PCTSP):

    # Stochastic variant of PCTSP, the real (stochastic) prize is only revealed when node is visited

    @staticmethod
    def get_costs(dataset, pi):
        return PCTSP._get_costs(dataset, pi, stochastic=True)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCTSP.initialize(*args, **kwargs, stochastic=True)


def generate_instance(size, penalty_factor=3):
    depot = torch.rand(2)
    loc = torch.rand(size, 2)

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[size] * (penalty_factor) / float(size)
    penalty = torch.rand(size) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = torch.rand(size) * 4 / float(size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = torch.rand(size) * deterministic_prize * 2

    return {
        'depot': depot,
        'loc': loc,
        'penalty': penalty,
        'deterministic_prize': deterministic_prize,
        'stochastic_prize': stochastic_prize
    }


# class PCTSPDataset(Dataset):

#     def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
#         super(PCTSPDataset, self).__init__()


#         # processed_tours = []
#         # max_len = max(tour.shape[0] - 1 for tour in self.tmp_solver.ref_tours)  # minus 1 for removed 0
#         # for tour in self.tmp_solver.ref_tours:
#         #     new_tour = tour[1:]  # remove the first 0
#         #     padded_tour = np.pad(new_tour, (0, max_len - new_tour.shape[0]), 'constant', constant_values=0)
#         #     processed_tours.append(padded_tour)
#         # self.ref_tours = torch.tensor(processed_tours, dtype=torch.long)


#     def __len__(self):

#     def __getitem__(self, idx):
#             'depots': self.depots[idx],
#             'points': self.points[idx],
#             'penalties': self.penalties[idx],
#             'norm_prizes': self.norm_prizes[idx],
#             # 'ref_tours': self.ref_tours[idx]
#         }

class PCTSPDataset(Dataset):
    """
    PCTSP Dataset with optional data augmentation.

    Args:
        filename (str): Path to the .txt file containing the PCTSP instance.
        size (int): Expected number of nodes in the problem (legacy, can be ignored if filename is provided).
        num_samples (int): Maximum number of instances loaded before augmentation.
        offset (int): First instance to load.
        distribution (any): Legacy argument, not used in this implementation.
        augment_mode (str, optional): The augmentation strategy.
                                      Can be '8fold', 'rotation', or 'hybrid'. Defaults to None.
        augment_params (dict, optional): Parameters for augmentation.
                                         For 'rotation' and 'hybrid', expects {'num_rotations': N}.
                                         Defaults to None.
    """

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None,
                 augment_mode=None, augment_num=None):
        super(PCTSPDataset, self).__init__()

        assert filename is not None, "Filename must be provided."
        assert os.path.splitext(filename)[1] == '.txt', "Invalid file format. Expected a ``.txt`` file."

        self.tmp_solver = PCTSPSolver()
        self.tmp_solver.from_txt(filename, ref=True)

        self.depots = torch.Tensor(self.tmp_solver.depots[offset:offset + num_samples])
        self.points = torch.Tensor(self.tmp_solver.points[offset:offset + num_samples])
        self.penalties = torch.Tensor(self.tmp_solver.penalties[offset:offset + num_samples])
        self.norm_prizes = torch.Tensor(self.tmp_solver.norm_prizes[offset:offset + num_samples])

        if augment_mode:
            self._augment_data(augment_mode, augment_num)

        self.size = self.depots.shape[0]

    def _augment_data(self, mode, augment_num):
        """Internal method to apply data augmentation."""
        if mode == '8fold':
            self.depots, self.points = self._augment_8_fold(self.depots, self.points)
        elif mode == 'rotation':
            self.depots, self.points = self._augment_rotation(self.depots, self.points, augment_num)
        elif mode == 'hybrid':
            self.depots, self.points = self._augment_hybrid(self.depots, self.points, augment_num)
        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

        # 奖惩项是与点绑定的，与坐标无关，因此直接复制即可
        num_augmentations = self.points.shape[0] // self.penalties.shape[0]
        self.penalties = self.penalties.repeat(num_augmentations, 1)
        self.norm_prizes = self.norm_prizes.repeat(num_augmentations, 1)

    @staticmethod
    def _augment_8_fold(depots, points):
        """Applies 8-fold augmentation (flips and rotations by 90 degrees)."""
        # 提取 x, y 坐标
        # depots shape: (batch, 2) -> (batch, 1, 2) to match points
        # points shape: (batch, problem_size, 2)
        depot_points = torch.cat([depots.unsqueeze(1), points], dim=1)

        x = depot_points[..., [0]]
        y = depot_points[..., [1]]

        # 8种变换
        dat1 = torch.cat((x, y), dim=-1)
        dat2 = torch.cat((1 - x, y), dim=-1)
        dat3 = torch.cat((x, 1 - y), dim=-1)
        dat4 = torch.cat((1 - x, 1 - y), dim=-1)
        dat5 = torch.cat((y, x), dim=-1)
        dat6 = torch.cat((1 - y, x), dim=-1)
        dat7 = torch.cat((y, 1 - x), dim=-1)
        dat8 = torch.cat((1 - y, 1 - x), dim=-1)

        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)

        # 分离 depots 和 points
        aug_depots = aug_problems[:, 0, :]
        aug_points = aug_problems[:, 1:, :]

        return aug_depots, aug_points

    @staticmethod
    def _augment_rotation(depots, points, num_rotations):
        """Applies rotation augmentation."""
        depot_points = torch.cat([depots.unsqueeze(1), points], dim=1)

        augmented = []
        angles = torch.linspace(0, 2 * math.pi, num_rotations)

        x = depot_points[..., 0]
        y = depot_points[..., 1]

        for angle in angles:
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)

            x_rotated = x * cos_theta - y * sin_theta
            y_rotated = x * sin_theta + y * cos_theta

            rotated = torch.stack((x_rotated, y_rotated), dim=-1)
            augmented.append(rotated)

        aug_problems = torch.cat(augmented, dim=0)

        aug_depots = aug_problems[:, 0, :]
        aug_points = aug_problems[:, 1:, :]

        return aug_depots, aug_points

    @staticmethod
    def _augment_hybrid(depots, points, num_rotations):
        """Applies hybrid augmentation (8-fold + extra rotations)."""
        assert num_rotations > 8, "For 'hybrid' mode, num_rotations must be greater than 8."

        # 先进行8-fold增强
        aug_depots_8fold, aug_points_8fold = PCTSPDataset._augment_8_fold(depots, points)

        depot_points = torch.cat([depots.unsqueeze(1), points], dim=1)

        # 计算还需要多少旋转
        num_extra_rotations = num_rotations - 8
        augmented = [torch.cat([aug_depots_8fold.unsqueeze(1), aug_points_8fold], dim=1)]

        # 避免与90度/180度/270度翻转重复，从一个小角度开始
        start_angle = 0.1
        angles = torch.linspace(start_angle, 2 * math.pi + start_angle, num_extra_rotations + 1)[:-1]

        x = depot_points[..., 0]
        y = depot_points[..., 1]

        for angle in angles:
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)

            x_rotated = x * cos_theta - y * sin_theta
            y_rotated = x * sin_theta + y * cos_theta

            rotated = torch.stack((x_rotated, y_rotated), dim=-1)
            augmented.append(rotated)

        aug_problems = torch.cat(augmented, dim=0)

        aug_depots = aug_problems[:, 0, :]
        aug_points = aug_problems[:, 1:, :]

        return aug_depots, aug_points

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'depots': self.depots[idx],
            'points': self.points[idx],
            'penalties': self.penalties[idx],
            'norm_prizes': self.norm_prizes[idx],
        }
