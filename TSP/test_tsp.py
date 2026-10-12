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
from TSPTester import TSPTester as Tester


##########################################################################################
# parameters
TEST_DATA = ('TSP', 'uniform', 50)
NODES_NUM = TEST_DATA[-1]
NUM_AUG = 16
USE_TWO_OPT = True

WEIGHT_DICT = {('TSP', 'uniform', 50): '../weights/tsp50.pt',
 ('TSP', 'uniform', 100): '../weights/tsp100.pt',
 ('TSP', 'uniform', 500): '../weights/tsp500.pt'}

TEST_PATH_DICT = {
    ("TSP", "uniform", 50): "../data/test/TSP/tsp50_concorde_5.68759.txt",
    ("TSP", "uniform", 100): "../data/test/TSP/tsp100_concorde_7.75585.txt",
    ("TSP", "uniform", 500): "../data/test/TSP/tsp500_concorde_16.54581.txt",
    }

NUM_INSTANCES_DICT = {
    ('TSP', 'uniform', 50): 1280,
    ('TSP', 'uniform', 100): 1280,
    ('TSP', 'uniform', 500): 128,
}

env_params = {
    "test_type": TEST_DATA,
    "node_cnt": NODES_NUM,
    "pomo_size": NODES_NUM,
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

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': WEIGHT_DICT[TEST_DATA],
    'test_file': TEST_PATH_DICT[TEST_DATA],
    'aug_factor': NUM_AUG,
    'use_2opt': USE_TWO_OPT,
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
    print("[Test Config] 2-OPT:", USE_TWO_OPT)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])


##########################################################################################

if __name__ == "__main__":
    main()
