
import torch
import numpy as np
import math

CAPACITY_DICT = {
    50: 12.5,
    100: 25,
    200: 25,
    500: 25,
}

def get_random_problems(batch_size, problem_size):
    weights = torch.rand(batch_size, problem_size + 1) # with dummy
    values = torch.rand(batch_size, problem_size + 1) # with dummy
    weights[:, -1] = 0
    values[:, -1] = 0
    capacities = torch.ones(batch_size) * CAPACITY_DICT[problem_size]
    pse_points = torch.stack([weights, values], dim=-1)
    return weights, values, capacities, pse_points


def augment_items(weights, values, capacities, aug_factor):
    """Return item permutations and maps from augmented indices to original items."""
    batch, width = weights.shape
    identity = torch.arange(width, device=weights.device).expand(batch, width)
    permutations = [identity]
    for _ in range(aug_factor - 1):
        real_items = torch.stack([
            torch.randperm(width - 1, device=weights.device) for _ in range(batch)
        ])
        dummy = torch.full((batch, 1), width - 1, dtype=torch.long, device=weights.device)
        permutations.append(torch.cat((real_items, dummy), dim=1))
    index_map = torch.cat(permutations, dim=0)
    aug_weights = weights.repeat(aug_factor, 1).gather(1, index_map)
    aug_values = values.repeat(aug_factor, 1).gather(1, index_map)
    return aug_weights, aug_values, capacities.repeat(aug_factor), index_map
