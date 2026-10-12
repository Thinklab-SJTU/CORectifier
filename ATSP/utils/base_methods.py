from ctypes import CDLL, c_int, c_double, byref
import numpy as np
import tsplib95
import lkh
import torch as th
from ml4co_kit import *
from typing import Union
from pathlib import Path

class BaseSolver:
    def __init__(self, n, lib_path=None, scaler=1e6):
        '''
        n int: number of nodes in the problems
        '''
        lib_path = lib_path or str(Path(__file__).with_name("libtsp.so"))
        self.lib_tsp = CDLL(lib_path)

        self.n = n
        self.path = (c_int * n)(*(list(range(n))))
        self.cost = c_double(0)
        self.scaler = scaler

    def _make_problem(self, dist):
        return (c_double * (self.n * self.n))(*dist.reshape(self.n * self.n).tolist())

    def get_cost(self):
        return self.cost.value

    def get_path(self):
        return list(self.path)

    def solve_random_walk(self, dist):
        pass

    def solve_nearest_neighbor(self, dist):
        '''
        dist numpy array: the distance matrix
        '''
        self.lib_tsp.nearest_neighbor(self.n,
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))

    def solve_nearest_insertion(self, dist):
        self.lib_tsp.nearest_insertion(self.n,
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))

    def solve_farthest_insertion(self, dist):
        self.lib_tsp.farthest_insertion(self.n,
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))

    def solve_random_insertion(self, distmap, order=None):
        distmap = to_numpy(distmap)
        order = list(range(len(distmap))) if order is None else to_numpy(order).tolist()
        tour = order[:2]
        for node in order[2:]:
            deltas = [distmap[tour[i], node] + distmap[node, tour[(i + 1) % len(tour)]]
                      - distmap[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))]
            tour.insert(int(np.argmin(deltas)) + 1, node)
        self.path = np.asarray(tour, dtype=np.int64)
        cost = float(distmap[self.path, np.roll(self.path, -1)].sum())
        return self.path, cost

    def solve_rand_perm(self, dist):
        self.path = np.random.permutation(self.n).tolist()

    def solve_lkh(self, dist, runs=1, max_trials=1000):
        import warnings
        from copy import deepcopy
        warnings.filterwarnings('ignore')
        def parse_tsp(X, name='unknown'):
            if isinstance(X, th.Tensor):
                X = X.cpu().numpy()
            self.n = X.shape[-1]
            scale_X = X.reshape((self.n, self.n)) * 1e6
            dim = len(scale_X)
            outstr = ''
            outstr += 'NAME: %s\n' % name #problem.name
            outstr += 'TYPE: ATSP\n'
            outstr += 'COMMENT: %s\n' % name
            outstr += 'DIMENSION: %d\n' % dim #problem.dimension
            outstr += 'EDGE_WEIGHT_TYPE: EXPLICIT\n'
            outstr += 'EDGE_WEIGHT_FORMAT: FULL_MATRIX\n' #LOWER_DIAG_ROW
            outstr += 'EDGE_WEIGHT_SECTION:\n'
            for l in scale_X:
                listToStr = ' '.join([str(elem) for elem in l])
                outstr += ' %s\n' % listToStr
            #outstr += 'EDGE_DATA_FORMAT: EDGE_LIST\n'
            #outstr += 'EDGE_DATA_SECTION:\n'
            #    outstr += f' {edge_idx[0]+1} {edge_idx[1]+1} {weight}\n'
            #outstr += '-1\n'
            return outstr
        def calc_lkh_tour_len(tsp, solver_path = "LKH"):
            def get_edge_weight(tsp, city1, city2):
                weight = tsp.get_weight(*(city1,city2))# if city1 > city2 else tsp.get_weight(*(city2,city1))
                return weight
            def calc_lkh_tour(tsp, solver_path):
                result= lkh.solve(solver_path, problem=tsp, runs=runs, max_trials=max_trials)
                lkh_path = result[0]
                lkh_path.append(lkh_path[0])
                return lkh_path
            lkh_path = calc_lkh_tour(tsp, solver_path)
            self.path = deepcopy(lkh_path)
            tour_len = 0
            while(len(lkh_path) > 1):
                start_node = lkh_path.pop()
                next_node = lkh_path[-1]
                tour_len += get_edge_weight(tsp, next_node-1, start_node-1)
            return tour_len
        tsp = tsplib95.parse(parse_tsp(dist))
        tour_len = calc_lkh_tour_len(tsp)
        tour_len *= 1e-6

        return tour_len

        # init path for trainning
        # result= lkh.solve('LKH', problem=tsp, runs=runs, max_trials=max_trials)
        # lkh_path = result[0]
        # lkh_path.append(lkh_path[0])
