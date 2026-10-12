##########################################################################################
# Machine Environment Config

USE_CUDA = True
USE_LABEL = True  # whether or not enable retifications
USE_IL = False
CUDA_DEVICE_NUM = 0

ITEMS_NUM = 50

# Appendix E, Table 15: Rectified RL training settings.
TRAIN_CONFIG_DICT = {
    50: {'epochs': 5000, 'batch_size': 200, 'samples_per_epoch': 10000, 'lr': 1e-4},
    100: {'epochs': 5000, 'batch_size': 200, 'samples_per_epoch': 10000, 'lr': 1e-4},
    200: {'epochs': 1000, 'batch_size': 64, 'samples_per_epoch': 10000, 'lr': 5e-5},
    500: {'epochs': 1000, 'batch_size': 16, 'samples_per_epoch': 10000, 'lr': 5e-5},
}
TRAIN_CONFIG = TRAIN_CONFIG_DICT[ITEMS_NUM]
MAX_EPOCHS = TRAIN_CONFIG['epochs']
BATCH_SIZE = TRAIN_CONFIG['batch_size']
SAMPLES_PER_EPOCH = TRAIN_CONFIG['samples_per_epoch']

MILESTONE_DICT = {
    5000: [4501, 4901],
    1000: [801, 901],
    500: [301, 401]
}

TRAIN_TYPE = ("KP", "uniform", ITEMS_NUM)

TRAIN_FOLDER_DICT = {
    ("KP", "uniform", 50): "../data/train/KP/KP50_Uniform_1.28m",
    ("KP", "uniform", 100): "../data/train/KP/KP100_Uniform_1.28m",
    ("KP", "uniform", 200): "../data/train/KP/KP200_Uniform_1.28m",
    ("KP", "uniform", 500): "../data/train/KP/KP500_Uniform_1.28m",
}

VAL_PATH_DICT = {
    ("KP", "uniform", 50): "../data/test/KP/kp50_uniform_c12.5_test_20.02065.txt",
    ("KP", "uniform", 100): "../data/test/KP/kp100_uniform_c25_test_40.30244.txt",
    ("KP", "uniform", 200): "../data/test/KP/kp200_uniform_c25_test_57.40207.txt",
    ("KP", "uniform", 500): "../data/test/KP/kp500_uniform_c25_test_91.12778.txt",
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
from KPTrainer import KPTrainer as Trainer
import warnings
warnings.filterwarnings("ignore")

##########################################################################################
# parameters

WEIGHT_DICT = {('KP', 'uniform', 50): '../weights/kp50.pt',
 ('KP', 'uniform', 100): '../weights/kp100.pt',
 ('KP', 'uniform', 200): '../weights/kp200.pt',
 ('KP', 'uniform', 500): '../weights/kp500.pt'}

env_params = {
    "pomo_size": min(ITEMS_NUM, 200), # if not USE_IL else 1,
    "problem_pool": [TRAIN_TYPE[0]],
    "item_cnt": ITEMS_NUM,
    "train_folder": TRAIN_FOLDER_DICT[TRAIN_TYPE],
    "use_label": USE_LABEL,
    "label_prob": 0.1,
    "instance_prob": 0.1,
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
        "lr": TRAIN_CONFIG['lr'],
        "weight_decay": 1e-6
    },
    "scheduler": {
        "milestones": MILESTONE_DICT.get(MAX_EPOCHS, [MAX_EPOCHS + 1]),
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
    'IL_loss': USE_IL,
    'aug_factor': 16,
}

logger_params = {
    "log_file": {
        "desc": f"train_{TRAIN_TYPE[0]}{ITEMS_NUM}_label={USE_LABEL}_IL={USE_IL}",
        "filename": "run_log.log"
    }
}


##########################################################################################
# main

def main():
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
