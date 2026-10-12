import sys
from pathlib import Path
import os
TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
os.chdir(TASK_DIR)
import sys

import os
import time
import torch
from run import run
from argparse import Namespace

opts = Namespace()

NODES_NUM = 20
TRAIN_TYPE = ("PCTSP", NODES_NUM)

TRAIN_FOLDER_DICT = {
    ("PCTSP", 20): "../data/train/PCTSP/pctsp20",
    ("PCTSP", 50): "../data/train/PCTSP/pctsp50",
    ("PCTSP", 100): "../data/train/PCTSP/pctsp100",
}

VAL_PATH_DICT = {
    ("PCTSP", 20): "../data/test/PCTSP/pctsp20_val.txt",
    ("PCTSP", 50): "../data/test/PCTSP/pctsp50_val.txt",
    ("PCTSP", 100): "../data/test/PCTSP/pctsp100_val.txt",
}


VAL_COST_DICT = {
    ("PCTSP", 20): "../data/test/PCTSP/pctsp20_cost.txt",
    ("PCTSP", 50): "../data/test/PCTSP/pctsp50_cost.txt",
    ("PCTSP", 100): "../data/test/PCTSP/pctsp100_cost.txt",
}


LOAD_PATH_DICT = {('PCTSP', 20): '../weights/pctsp20.pt',
 ('PCTSP', 50): '../weights/pctsp50.pt',
 ('PCTSP', 100): '../weights/pctsp100.pt'}


# Rectify
opts.use_label = False
opts.il_loss = False
opts.label_prob = 0.1  # 0.5
opts.instance_prob = 0.1  # 0.5
opts.act_prob = 1.0
opts.rect_range = [0.1, 0.2]  # [0.8, 1.0]

# Env
opts.problem = TRAIN_TYPE[0].lower()
opts.graph_size = TRAIN_TYPE[1]
opts.batch_size = 1
opts.epoch_size = 1
opts.val_size = 2
opts.val_dataset = VAL_PATH_DICT[TRAIN_TYPE]
opts.train_folder = TRAIN_FOLDER_DICT[TRAIN_TYPE]
opts.val_cost_path = VAL_COST_DICT[TRAIN_TYPE]

# Model
opts.model = 'attention'
opts.embedding_dim = 128
opts.hidden_dim = 128
opts.n_encode_layers = 3
opts.tanh_clipping = 10.
opts.normalization = 'batch'

# Training
opts.lr_model = 1e-4
opts.lr_critic = 1e-4
opts.lr_decay = 1.0
opts.eval_only = True
opts.n_epochs = 1
opts.seed = 1234
opts.max_grad_norm = 1.0
opts.no_cuda = False
opts.exp_beta = 0.8
opts.baseline = 'rollout'
opts.bl_alpha = 0.05
opts.bl_warmup_epochs = None
opts.eval_batch_size = 16
opts.checkpoint_encoder = False
opts.shrink_size = None
opts.data_distribution = None
opts.augment_mode = 'hybrid'
opts.augment_num = 16

# Misc
opts.log_step = 50
opts.log_dir = 'logs'
opts.run_name = 'run'
opts.output_dir = 'outputs'
opts.epoch_start = 0
opts.checkpoint_epochs = 1
opts.load_path = LOAD_PATH_DICT[TRAIN_TYPE]
opts.resume = None
opts.no_tensorboard = True
opts.no_progress_bar = False


opts.cuda_device_num = 0
opts.single_gpu = True

opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
opts.save_dir = os.path.join(
    opts.output_dir,
    "{}_{}_{}".format(opts.graph_size, opts.use_label, opts.il_loss),
    opts.run_name
)
if opts.bl_warmup_epochs is None:
    opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"

if __name__ == "__main__":
    run(opts)
