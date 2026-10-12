"""Compatibility with data-reader objects embedded in original PCTSP checkpoints."""
import io
import pickle
from enum import Enum
from pickle import dump, dumps

from utils.pctsp_data import PCTSPSolver


class _LegacyTaskType(str, Enum):
    PCTSP = "Prize Collection Traveling Salesman Problem (PCTSP)"


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("ml4co_kit.solver.pctsp.base", "PCTSPSolver"):
            return PCTSPSolver
        if (module, name) == ("ml4co_kit.utils.type_utils", "TASK_TYPE"):
            return _LegacyTaskType
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def loads(data, **kwargs):
    return load(io.BytesIO(data), **kwargs)
