import numpy as np
import torch as th
import scipy.spatial as ssp

def count_nodes(num_clauses, num_variables):
    return 2 * num_clauses * num_variables + num_clauses

def gen_3sat(num_clauses, num_variables):
    """ randomly generate 3sat distance matries """
    assert num_variables>=3, "num_variables should be no smaller than 3"

    num_nodes = count_nodes(num_clauses, num_variables)
    dist = np.ones((num_nodes, num_nodes))

    # varialble node [0, 2 * num_clauses * num_variables-1]
    for v in range(num_variables):
        ofs = v * 2 * num_clauses
        for c in range(num_clauses):
            dist[ofs + 2 * c, ofs + 2 * c + 1] = 0
            dist[ofs + 2 * c + 1, ofs + 2 * c] = 0
            if c != num_clauses - 1:
                dist[ofs + 2 * c + 1, ofs + 2 * c + 2] = 0
                dist[ofs + 2 * c + 2, ofs + 2 * c + 1] = 0
        dist[ofs, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
        dist[ofs, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0
        dist[ofs + 2 * num_clauses - 1, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
        dist[ofs + 2 * num_clauses - 1, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0

    # clause node [2 * num_clauses, 2 * num_clauses + num_clauses]
    ofs_clause = 2 * num_clauses * num_variables
    for c in range(num_clauses):
        # sampling variables
        vars = np.random.choice(num_variables, size=3, replace=False)
        # sampling signs
        signs = np.random.choice(2, 3,replace=True)
        for i in range(3):
            ofs_var = vars[i] * 2 * num_clauses
            if signs[i] == 0: # x
                dist[ofs_var + 2 * c, ofs_clause + c] = 0
                dist[ofs_clause + c, ofs_var + 2 * c + 1] = 0
            else: # not x
                dist[ofs_var + 2 * c + 1, ofs_clause + c] = 0
                dist[ofs_clause + c, ofs_var + 2 * c] = 0
    return dist

def gen_hcp(num_nodes, noise_level):
    dist = th.ones((num_nodes, num_nodes))
    hpath = th.randperm(num_nodes)
    dist[hpath, hpath.roll(-1)] = 0
    num_noise_edges = int(noise_level * num_nodes * num_nodes)
    if num_noise_edges > 0:
        heads = th.LongTensor(np.random.choice(num_nodes, size=num_noise_edges, replace=True))
        tails = th.LongTensor(np.random.choice(num_nodes, size=num_noise_edges, replace=True))
        dist[heads, tails] = 0
    return dist.cpu().numpy()

def gen_Euclidean(n, d):
    position = np.random.rand(n, d)
    dist = ssp.distance_matrix(position, position)
    return dist


