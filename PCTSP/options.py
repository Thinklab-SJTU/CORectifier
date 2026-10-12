import os
import time
import argparse
import torch

# Appendix E, Table 14: Rectified RL training settings.
TRAIN_CONFIG_DICT = {
    20: {'epochs': 10000, 'batch_size': 64, 'samples_per_epoch': 6400, 'lr': 1e-4},
    50: {'epochs': 10000, 'batch_size': 64, 'samples_per_epoch': 6400, 'lr': 1e-4},
    100: {'epochs': 10000, 'batch_size': 64, 'samples_per_epoch': 6400, 'lr': 1e-4},
}


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='pctsp', help="The problem to solve, default 'pctsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size; defaults to Appendix E for graph_size')
    parser.add_argument('--epoch_size', type=int, default=None, help='Instances per epoch; defaults to Appendix E for graph_size')
    parser.add_argument('--val_size', type=int, default=2,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Validation dataset; defaults to the selected graph_size')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=None, help="Actor learning rate; defaults to Appendix E for graph_size")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Adam weight decay (Appendix E)')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=None, help='Training epochs; defaults to Appendix E for graph_size')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to rollout baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=0,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    parser.add_argument('--train_folder', default=None)
    parser.add_argument('--val_cost_path', default=None)
    parser.add_argument('--no_label', dest='use_label', action='store_false')
    parser.add_argument('--il_loss', action='store_true')
    parser.add_argument('--label_prob', type=float, default=0.1)
    parser.add_argument('--instance_prob', type=float, default=0.1)
    parser.add_argument('--act_prob', type=float, default=1.0)
    parser.add_argument('--rect_range', type=float, nargs=2, default=[0.1, 0.2])
    parser.add_argument('--augment_mode', default='hybrid')
    parser.add_argument('--augment_num', type=int, default=16)
    parser.add_argument('--cuda_device_num', type=int, default=0)
    parser.add_argument('--multi_gpu', dest='single_gpu', action='store_false')
    parser.set_defaults(use_label=True, single_gpu=True, no_tensorboard=True)
    opts = parser.parse_args(args)

    if opts.graph_size not in TRAIN_CONFIG_DICT:
        parser.error('PCTSP graph_size must be one of 20, 50, 100')
    train_config = TRAIN_CONFIG_DICT[opts.graph_size]
    for option, setting in [('n_epochs', 'epochs'), ('batch_size', 'batch_size'),
                            ('epoch_size', 'samples_per_epoch'), ('lr_model', 'lr')]:
        if getattr(opts, option) is None:
            setattr(opts, option, train_config[setting])
    if opts.val_dataset is None:
        opts.val_dataset = f'../data/test/PCTSP/pctsp{opts.graph_size}_val.txt'
    if opts.train_folder is None:
        opts.train_folder = f'../data/train/PCTSP/pctsp{opts.graph_size}'
    if opts.val_cost_path is None:
        opts.val_cost_path = f'../data/test/PCTSP/pctsp{opts.graph_size}_cost.txt'

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
