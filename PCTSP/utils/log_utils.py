def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.add_scalar('avg_cost', avg_cost, step)

        tb_logger.add_scalar('actor_loss', reinforce_loss.item(), step)
        tb_logger.add_scalar('nll', -log_likelihood.mean().item(), step)

        tb_logger.add_scalar('grad_norm', grad_norms[0], step)
        tb_logger.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.add_scalar('critic_loss', bl_loss.item(), step)
            tb_logger.add_scalar('critic_grad_norm', grad_norms[1], step)
            tb_logger.add_scalar('critic_grad_norm_clipped', grad_norms_clipped[1], step)


import os
from datetime import datetime

def log_to_file(message, opts):
    """
    Appends a message to a log file within the specified save directory.
    The log file will be named 'validation_log.txt'.

    Args:
        message (str): The string message to be logged.
        opts (Namespace): The options object which must contain 'save_dir'.
    """
    if not os.path.isdir(opts.save_dir):
        print(f"Warning: Log directory {opts.save_dir} not found. Cannot write log.")
        return

    log_filepath = os.path.join(opts.save_dir, 'validation_log.txt')

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    try:
        with open(log_filepath, 'a') as f:
            f.write(log_entry)

    except IOError as e:
        print(f"Error: Could not write to log file {log_filepath}. Reason: {e}")
