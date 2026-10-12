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

from ATSPTester import ATSPTester as Tester


##########################################################################################
# parameters

TEST_DATA = ('ATSP', 'uniform', 50)
NODES_NUM = TEST_DATA[-1]
NUM_AUG = 16
USE_TWO_OPT = True

WEIGHT_DICT = {('ATSP', 'uniform', 50): '../weights/atsp50.pt',
 ('ATSP', 'uniform', 100): '../weights/atsp100.pt',
 ('ATSP', 'uniform', 200): '../weights/atsp200.pt',
 ('ATSP', 'uniform', 500): '../weights/atsp500.pt'}

TEST_PATH_DICT = {
    ("ATSP", "uniform", 50): "../data/test/ATSP/atsp50_uniform_lkh_1000_1.55448.txt",
    ("ATSP", "uniform", 100): "../data/test/ATSP/atsp100_uniform_lkh_1000_1.56604.txt",
    ("ATSP", "uniform", 200): "../data/test/ATSP/atsp200_uniform_lkh_1000_1.56472.txt",
    ("ATSP", "uniform", 500): "../data/test/ATSP/atsp500_uniform_lkh_1000_1.57345.txt",
}

env_params = {
    "test_type": TEST_DATA,
    "init_solver": "nn",
    "pos_embedding_dim": 256,
    "node_cnt": TEST_DATA[2],
    "pomo_size": NODES_NUM,
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
    "eval_type": "softmax"
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': WEIGHT_DICT[TEST_DATA],
    'test_file': TEST_PATH_DICT[TEST_DATA],
    'augmentation_enable': True,
    'aug_factor': NUM_AUG,
    'use_2opt': USE_TWO_OPT,
    'num_instances': 1,
    'rectify_ratio': 0.1,
    'rectify_lens': 1,
    'rectify_iters': 1,
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
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] 2-OPT:", USE_TWO_OPT)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])


##########################################################################################

if __name__ == "__main__":
    main()
