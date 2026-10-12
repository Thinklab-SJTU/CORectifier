##########################################################################################
# Machine Environment Config

USE_CUDA = True
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import torch
import numpy as np
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))


##########################################################################################
# import

import logging
from KPTester import KPTester as Tester


##########################################################################################
# parameters
TEST_DATA = ('KP', 'uniform', 50)
NODES_NUM = TEST_DATA[-1]
NUM_AUG = 16

WEIGHT_DICT = {('KP', 'uniform', 50): '../weights/kp50.pt',
 ('KP', 'uniform', 100): '../weights/kp100.pt',
 ('KP', 'uniform', 200): '../weights/kp200.pt',
 ('KP', 'uniform', 500): '../weights/kp500.pt'}

TEST_PATH_DICT = {
    ("KP", "uniform", 50): "../data/test/KP/kp50_uniform_c12.5_test_20.02065.txt",
    ("KP", "uniform", 100): "../data/test/KP/kp100_uniform_c25_test_40.30244.txt",
    ("KP", "uniform", 200): "../data/test/KP/kp200_uniform_c25_test_57.40207.txt",
    ("KP", "uniform", 500): "../data/test/KP/kp500_uniform_c25_test_91.12778.txt",
}

NUM_INSTANCES_DICT = {
    ('KP', 'uniform', 50): 1280,
    ('KP', 'uniform', 100): 1280,
    ('KP', 'uniform', 200): 1280,
    ('KP', 'uniform', 500): 1280,
}

env_params = {
    "test_type": TEST_DATA,
    "item_cnt": NODES_NUM,
    "pomo_size": NODES_NUM,
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

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': WEIGHT_DICT[TEST_DATA],
    'test_file': TEST_PATH_DICT[TEST_DATA],
    'aug_factor': NUM_AUG,
    'num_instances': NUM_INSTANCES_DICT[TEST_DATA],
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

##########################################################################################
# main

def main():
    setup_seed(1234)

    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    tester.run()

def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] Instances:", tester_params['num_instances'])
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])


##########################################################################################

if __name__ == "__main__":
    main()
