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
TEST_DIR = '../data/test/CVRPLIB'
NUM_AUG = 16
USE_LS = True

TEST_PATH_LIST = []
TEST_NODE_CNT_LIST = []

for file in sorted(os.listdir(TEST_DIR)):
    test_txt = os.path.join(TEST_DIR, file)
    num_nodes = int(file.split("-")[1][1:]) - 1
    TEST_PATH_LIST.append(test_txt)
    TEST_NODE_CNT_LIST.append(int(num_nodes))

WEIGHT_DICT = {('CVRP', 'uniform', 50): '../weights/cvrp50.pt',
 ('CVRP', 'uniform', 100): '../weights/cvrp100_cvrplib.pt',
 ('CVRP', 'uniform', 200): '../weights/cvrp200.pt',
 ('CVRP', 'uniform', 500): '../weights/cvrp500.pt'}


env_params = {
    "test_type": TEST_DATA,
    "node_cnt": TEST_DATA[-1],
    "pomo_size": TEST_DATA[-1],
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
    'test_file': None,
    'aug_factor': NUM_AUG,
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
    objs_list = []
    time_list = []

    for inst, num_nodes in zip(TEST_PATH_LIST, TEST_NODE_CNT_LIST):
        env_params = {
            "test_type": TEST_DATA,
            "node_cnt": num_nodes,
            "pomo_size": num_nodes,
        }
        tester_params["test_file"] = inst
        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
        metrics, inst_time = tester.run()
        gaps_list.append(metrics[2])
        time_list.append(inst_time)
        objs_list.append(metrics[0])

    print(f"[Test Result] Avg. obj : {sum(objs_list) / len(objs_list):.5f}")
    print(f"[Test Result] Avg. gap : {sum(gaps_list) / len(gaps_list):.5f}%")
    print(f"[Test Result] Avg. time: {sum(time_list) / len(time_list):.5f}s")

def _print_config():
    print("[Test Config] Dataset:", TEST_DATA)
    print("[Test Config] N_Aug:", NUM_AUG)
    print("[Test Config] Local Search:", USE_LS)
    print("[Test Config] Weight:", WEIGHT_DICT[TEST_DATA])

##########################################################################################

if __name__ == "__main__":
    main()
