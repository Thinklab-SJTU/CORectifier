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

from CVRPTester import CVRPTester as Tester


##########################################################################################
# parameters
TEST_DATA = ('CVRP', 'uniform', 50)
NODES_NUM = TEST_DATA[-1]
NUM_AUG = 16
USE_LS = True

WEIGHT_DICT = {('CVRP', 'uniform', 50): '../weights/cvrp50.pt',
 ('CVRP', 'uniform', 100): '../weights/cvrp100.pt',
 ('CVRP', 'uniform', 200): '../weights/cvrp200.pt',
 ('CVRP', 'uniform', 500): '../weights/cvrp500.pt'}


TEST_PATH_DICT = {
    ("CVRP", "uniform", 50): "../data/test/CVRP/cvrp50_hgs-1s_10.366.txt",
    ("CVRP", "uniform", 100): "../data/test/CVRP/cvrp100_hgs-20s_15.563.txt",
    ("CVRP", "uniform", 200): "../data/test/CVRP/cvrp200_hgs-60s_19.630.txt",
    ("CVRP", "uniform", 500): "../data/test/CVRP/cvrp500_hgs-300s_37.154.txt",
}

NUM_INSTANCES_DICT = {
    ('CVRP', 'uniform', 50): 10000,
    ('CVRP', 'uniform', 100): 10000,
    ('CVRP', 'uniform', 200): 100,
    ('CVRP', 'uniform', 500): 100,
}

env_params = {
    "test_type": TEST_DATA,
    "node_cnt": NODES_NUM,
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
    'use_ls': USE_LS,
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

    tester = Tester(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params
    )
    tester.run()

def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] Instances:", tester_params['num_instances'])
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] Local Search:", USE_LS)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])

##########################################################################################

if __name__ == "__main__":
    main()
