"""Classic ML4CO/HGS local search with buffers sized for untrained policies."""
import ctypes
from pathlib import Path

import numpy as np


_LIB = None


def cvrp_classic_local_search(init_tour, depot, points, demands,
                             coords_scale=1000, demands_scale=1000, seed=1234):
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(Path(__file__).parent / "c_local_search/cvrp_local_search.so"))
        _LIB.cvrp_local_search.argtypes = [
            ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), *([ctypes.c_int] * 5)
        ]
        _LIB.cvrp_local_search.restype = ctypes.POINTER(ctypes.c_int)
        _LIB.free_cvrp_result.argtypes = [ctypes.POINTER(ctypes.c_int)]
        _LIB.free_cvrp_result.restype = None

    tour = np.asarray(init_tour, dtype=np.int16)
    depot = np.asarray(depot)
    points = np.asarray(points)
    demands = np.asarray(demands)
    if tour.ndim != 1 or depot.shape != (2,) or points.ndim != 2 or demands.ndim != 1:
        raise ValueError("Expected a 1D tour, a depot, customer coordinates and demands.")
    if tour[0] != 0 or tour[-1] != 0 or sorted(tour[tour > 0].tolist()) != list(range(1, len(points) + 1)):
        raise ValueError("The initial tour must visit each customer once and start/end at the depot.")
    if len(demands) == len(points):
        demands = np.insert(demands, 0, 0)
    coords = np.ascontiguousarray(np.concatenate((depot[None, :], points)), dtype=np.float32)
    demands = np.ascontiguousarray(demands, dtype=np.float32)
    tour = np.ascontiguousarray(tour)
    result = _LIB.cvrp_local_search(
        tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        demands.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(points) + 1, len(tour), coords_scale, demands_scale, seed
    )
    try:
        if result[0] == -1:
            return tour.astype(np.int64)
        length = 0
        while result[length] != -1:
            length += 1
        return np.ctypeslib.as_array(result, shape=(length,)).copy()
    finally:
        _LIB.free_cvrp_result(result)
