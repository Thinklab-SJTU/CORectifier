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
import re

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

TEST_PATH_LIST = []
TEST_NODE_CNT_LIST = []
for file in sorted(os.listdir("../data/test/TSPLIB")):
    test_txt = os.path.join("../data/test/TSPLIB", file)
    TEST_PATH_LIST.append(test_txt)

    pattern = r"[a-zA-Z]+\d+"
    match = re.search(pattern, file)
    if match:
        alphanumeric = match.group()
        num_nodes = re.sub(r"[a-zA-Z]", "", alphanumeric)
    TEST_NODE_CNT_LIST.append(int(num_nodes))

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

# model_params = {

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': WEIGHT_DICT[TEST_DATA],
    'test_file': None,
    'aug_factor': NUM_AUG,
    'use_2opt': USE_TWO_OPT,
    'num_instances': 1,
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

    gaps_list = []

    for inst, num_nodes in zip(TEST_PATH_LIST, TEST_NODE_CNT_LIST):
        print(inst.split("/")[-1])
        env_params = {
            "test_type": TEST_DATA,
            "node_cnt": num_nodes,
            "pomo_size": num_nodes,
        }
        tester_params["test_file"] = inst
        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
        metrics = tester.run()
        gaps_list.append(metrics[2])
    print(f"avg. gap over {len(gaps_list)}: {sum(gaps_list) / len(gaps_list)}%")

def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] 2-OPT:", USE_TWO_OPT)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])


##########################################################################################

if __name__ == "__main__":
    main()
