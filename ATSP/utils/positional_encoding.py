"""Positional features used by the ATSP encoder.

The encoder receives a cyclic hyperbolic positional basis for every node.
"""

import torch


def make_positional_encoding_cosh_recur(dim, n_node, scaler=10):
    """Build the cyclic reciprocal-cosh positional encoding used by ATSP."""
    dim_phi = torch.arange(0, dim, dtype=torch.float) / dim * 2 - 1
    position_phi = torch.arange(0, n_node, dtype=torch.float).unsqueeze(1) / n_node * 2
    phi = dim_phi + position_phi
    phi[phi > 1] -= 2
    return 1 / torch.cosh(phi * scaler)
