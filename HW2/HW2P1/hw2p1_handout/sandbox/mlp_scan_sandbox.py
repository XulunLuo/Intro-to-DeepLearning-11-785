import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Absolute path to the root directory
mytorch_dir = os.path.join(project_root, 'mytorch')
mytorch_nn_dir = os.path.join(mytorch_dir, 'nn')
models_dir = os.path.join(project_root, 'models')

sys.path.append(mytorch_dir)
sys.path.append(mytorch_nn_dir)
sys.path.append(models_dir)

from mlp_scan import *
from itertools import product

# Note: Feel free to change anything about this file for your testing purposes

# There's not much need for a sandbox for this model, but if you would like to test your model, you may do so here.

# Load the autograder data
data = np.loadtxt(os.path.join(project_root, 'autograder', 'data', 'data.asc')).T.reshape(1, 24, -1)
print("Input shape:", data.shape)

weights_c = np.load(os.path.join(project_root, 'autograder', 'weights', 'mlp_weights_part_c.npy'), allow_pickle=True)
expected_distributed = np.load(os.path.join(project_root, 'autograder', 'ref_result', 'res_c.npy'), allow_pickle=True)

cnn_dist = CNN_DistributedScanningMLP()
cnn_dist.init_weights(weights_c)
result = cnn_dist.forward(data)
print("Output shape:", result.shape)
print("Match:", np.allclose(result, expected_distributed))