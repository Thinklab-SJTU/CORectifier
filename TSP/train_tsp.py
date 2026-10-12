##########################################################################################
# Machine Environment Config

USE_CUDA = True
USE_LABEL = True  # whether or not enable retifications
CUDA_DEVICE_NUM = 0

NODES_NUM = 50
MAX_EPOCHS = 1
BATCH_SIZE = 1
SAMPLES_PER_EPOCH = 1

TRAIN_TYPE = ("TSP", "uniform", NODES_NUM)

TRAIN_FOLDER_DICT = {
    ("TSP", "uniform", 50): "../data/train/TSP/TSP50_Uniform_Concorde",
    ("TSP", "uniform", 100): "../data/train/TSP/TSP100_Uniform_Concorde",
    ("TSP", "uniform", 500):"../data/train/TSP/TSP500_Uniform_Concorde",
}

VAL_PATH_DICT = {
    ("TSP", "uniform", 50): "../data/test/TSP/tsp50_concorde_5.68759.txt",
    ("TSP", "uniform", 100): "../data/test/TSP/tsp100_concorde_7.75585.txt",
    ("TSP", "uniform", 500): "../data/test/TSP/tsp500_concorde_16.54581.txt",
}

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))


##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src
from TSPTrainer import TSPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

##########################################################################################
# parameters

WEIGHT_DICT = {('TSP', 'uniform', 50): '../weights/tsp50.pt',
 ('TSP', 'uniform', 100): '../weights/tsp100.pt',
 ('TSP', 'uniform', 500): '../weights/tsp500.pt'}

env_params = {
    "max_scale": NODES_NUM + 1,
    "min_scale": NODES_NUM,
    "pomo_size": NODES_NUM if NODES_NUM <= 200 else 200,
    "problem_pool": [TRAIN_TYPE[0]],
    "node_cnt": NODES_NUM,
    "problem_gen_params": {
        "int_min": 0,
        "int_max": 1000*1000,
        "scaler": 1000*1000
    },
    "train_folder": TRAIN_FOLDER_DICT[TRAIN_TYPE],
    "use_label": USE_LABEL,
    "label_prob": 0.1, # 0.5 0.25 0.1
    "instance_prob": 0.1, # 1. for pure IL
    "act_prob": 1.,
    "rect_range": [0.1, 0.2], # [0.8, 0.85]
}

model_params = {
    "embedding_dim": 256,
    "sqrt_embedding_dim": 256**(1/2),
    "encoder_layer_num": 5,
    "qkv_dim": 16,
    "sqrt_qkv_dim": 16**(1/2),
    "head_num": 16,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "ms_layer1_init": (1/2)**(1/2),
    "ms_layer2_init": (1/16)**(1/2),
    "eval_type": "argmax",
}

optimizer_params = {
    "type": "multi-step", # "cosine" if USE_LABEL else
    "optimizer": {
        "lr": 1e-4 if NODES_NUM > 50 else 2e-4,
        "weight_decay": 1e-6
    },
    "scheduler": {
        "milestones": [2001, 2101] if NODES_NUM <= 100 else [1001, 1101],  # if further training is needed
        "gamma": 0.1,
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': MAX_EPOCHS,
    'train_episodes': SAMPLES_PER_EPOCH,
    'train_batch_size': BATCH_SIZE,
    'val_interval': 1,
    'val_file': VAL_PATH_DICT[TRAIN_TYPE],
    'val_num': 1,
    'logging': {'model_save_interval': 1},
    'model_load': {'enable': False, 'only_weight': True, 'path': WEIGHT_DICT[TRAIN_TYPE], 'epoch': 0},
    'IL_loss': False,
    'aug_factor': 16,
    'use_2opt': True,
}

logger_params = {
    "log_file": {
        "desc": f"train_{TRAIN_TYPE[0]}{NODES_NUM}_{USE_LABEL}",
        "filename": "log.txt"
    }
}


##########################################################################################
# main
import random
import torch
import numpy as np

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    #     _set_debug_mode()
    seed_everything(3)

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _print_config():
    logger = logging.getLogger("root")
    logger.info("USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith("params")]


##########################################################################################

if __name__ == "__main__":
    main()
