##########################################################################################
# Machine Environment Config

USE_CUDA = True
USE_LABEL = True  # enable retification
USE_IL = False
CUDA_DEVICE_NUM = 0

NODES_NUM = 50
MAX_EPOCHS = 1
BATCH_SIZE = 1
#Batch_size for RL training: 200 for n=50, 128 for n=100, 16 for n=200, 4 for 500
#Batch_size for IL training (n* aug, pomo=1): 64 for n=50, 32 for n=100, 16 for n=200
SAMPLES_PER_EPOCH = 1

TRAIN_TYPE = ("CVRP", "uniform", NODES_NUM)

TRAIN_FOLDER_DICT = {
    ("CVRP", "uniform", 50): "../data/train/CVRP/CVRP50_Uniform_C40_HGS-1s",
    ("CVRP", "uniform", 100): "../data/train/CVRP/CVRP100_Uniform_C50_HGS-20s",
    ("CVRP", "uniform", 200): "../data/train/CVRP/cvrp200_train",
    ("CVRP", "uniform", 500): "../data/train/CVRP/cvrp500_train"
}

VAL_PATH_DICT = {
    ("CVRP", "uniform", 50): "../data/test/CVRP/cvrp50_hgs-1s_10.366.txt",
    ("CVRP", "uniform", 100): "../data/test/CVRP/cvrp100_hgs-20s_15.563.txt",
    ("CVRP", "uniform", 200): "../data/test/CVRP/cvrp200_hgs-60s_19.630.txt",
    ("CVRP", "uniform", 500): "../data/test/CVRP/cvrp500_hgs-300s_37.154.txt",
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

from CVRPTrainer import CVRPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

##########################################################################################
# parameters

WEIGHT_DICT = {('CVRP', 'uniform', 50): '../weights/cvrp50.pt',
 ('CVRP', 'uniform', 100): '../weights/cvrp100.pt',
 ('CVRP', 'uniform', 200): '../weights/cvrp200.pt',
 ('CVRP', 'uniform', 500): '../weights/cvrp500.pt'}

env_params = {
    'pomo_size': min(NODES_NUM, 200) if not USE_IL else 1,
    'problem_pool': [TRAIN_TYPE[0]],
    'node_cnt': NODES_NUM,
    'train_folder': TRAIN_FOLDER_DICT[TRAIN_TYPE],
    'use_label': USE_LABEL,
    'label_prob': 0.1,
    'instance_prob': 0.1,
    'act_prob': 0.1,
}

model_params = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128**(1/2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}

optimizer_params = {
    "type": "multi-step", # "cosine" if USE_LABEL else
    "optimizer": {
        "lr": 2e-4 if NODES_NUM <= 100 else 1e-4,
        "weight_decay": 1e-6
    },
    "scheduler": {
        "milestones": [1601, 1701],
        # [1501, 1901] for IL [2001, 2101] for RL n=50/100
        "gamma": 0.1
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
    'prev_model_path': None,
    'logging': {'model_save_interval': 1},
    'model_load': {'enable': False, 'only_weight': True, 'path': WEIGHT_DICT[TRAIN_TYPE], 'epoch': 0},
    'IL_loss': USE_IL,
    'aug_factor': 16,
    'use_ls': True,
}

logger_params = {
    "log_file": {
        "desc": f"train_{TRAIN_TYPE[0]}{NODES_NUM}_label={USE_LABEL}_IL={USE_IL}",
        "filename": "run_log.log"
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
