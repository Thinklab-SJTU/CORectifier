##########################################################################################
# Machine Environment Config

USE_CUDA = True
USE_LABEL = True  # whether or not enable retifications
CUDA_DEVICE_NUM = 0

NODES_NUM = 50

# Appendix E, Table 14: Rectified RL training settings.
TRAIN_CONFIG_DICT = {
    50: {'epochs': 8000, 'batch_size': 200, 'samples_per_epoch': 10000, 'lr': 4e-4},
    100: {'epochs': 3000, 'batch_size': 64, 'samples_per_epoch': 10000, 'lr': 2e-4},
    200: {'epochs': 3000, 'batch_size': 8, 'samples_per_epoch': 10000, 'lr': 2e-4},
    500: {'epochs': 2000, 'batch_size': 2, 'samples_per_epoch': 10000, 'lr': 5e-5},
}
TRAIN_CONFIG = TRAIN_CONFIG_DICT[NODES_NUM]
MAX_EPOCHS = TRAIN_CONFIG['epochs']
BATCH_SIZE = TRAIN_CONFIG['batch_size']
SAMPLES_PER_EPOCH = TRAIN_CONFIG['samples_per_epoch']

TRAIN_TYPE = ("ATSP", "uniform", NODES_NUM)

TRAIN_FOLDER_DICT = {
    ("ATSP", "uniform", 50): "../data/train/ATSP/ATSP50_Uniform_LKH-500",
    ("ATSP", "uniform", 100): "../data/train/ATSP/ATSP100_Unifrom_LKH-500",
    ("ATSP", "uniform", 200): "../data/train/ATSP/ATSP200_Uniform_LKH-1000",
    ("ATSP", "uniform", 500): "../data/train/ATSP/atsp500_train",
    ("TSP", "uniform", 50): "../data/train/TSP/TSP50_Uniform_Concorde",
    ("TSP", "uniform", 100): "../data/train/TSP/TSP100_Uniform_Concorde",
    ("TSP", "uniform", 500):"../data/train/TSP/TSP500_Uniform_Concorde",
}

VAL_PATH_DICT = {
    ("ATSP", "uniform", 50): "../data/test/ATSP/atsp50_uniform_lkh_1000_1.55448.txt",
    ("ATSP", "uniform", 100): "../data/test/ATSP/atsp100_uniform_lkh_1000_1.56604.txt",
    ("ATSP", "uniform", 200): "../data/test/ATSP/atsp200_uniform_lkh_1000_1.56472.txt",
    ("ATSP", "uniform", 500): "../data/test/ATSP/atsp500_uniform_lkh_1000_1.57345.txt",
}

##########################################################################################
# Path Config

import os
import sys

__file__ = os.path.abspath(__file__)
os.chdir(os.path.dirname(__file__))


##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src
from ATSPTrainer import ATSPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

##########################################################################################
# parameters

WEIGHT_DICT = {('ATSP', 'uniform', 50): '../weights/atsp50.pt',
 ('ATSP', 'uniform', 100): '../weights/atsp100.pt',
 ('ATSP', 'uniform', 200): '../weights/atsp200.pt',
 ('ATSP', 'uniform', 500): '../weights/atsp500.pt'}

env_params = {
    "max_scale": NODES_NUM + 1,
    "min_scale": NODES_NUM,
    "pomo_size": NODES_NUM if NODES_NUM <= 200 else 200,
    "problem_pool": [TRAIN_TYPE[0]],
    "init_solver": "nn" if USE_LABEL else None,
    "pos_embedding_dim": 256,
    "node_cnt": NODES_NUM,
    "problem_gen_params": {
        "int_min": 0,
        "int_max": 1000*1000,
        "scaler": 1000*1000
    },
    "train_folder": TRAIN_FOLDER_DICT[TRAIN_TYPE],
    "use_label": USE_LABEL,
    "label_prob": 0.5, # 0.5 0.25 0.1
    "instance_prob": 0.5, # 1. for pure IL
    "act_prob": 1.,
    "rect_range": [0.8, 1.0], # [0.8, 0.85]
}

model_params = {
    "pos_embedding_dim": env_params["pos_embedding_dim"],
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
        "lr": TRAIN_CONFIG['lr'],
        "weight_decay": 1e-6
    },
    "scheduler": {
        "milestones": [2001, 2101], # if NODES_NUM <= 100 else [1001, 1101],  # if further training is needed
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
    seed_everything(2)

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
