# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class LayerNorm1d:
    def __init__(self, num_features, eps=1e-8):
        self.num_features = num_features
        self.eps = eps

        self.gamma = np.ones((1, num_features), dtype=np.float32)  # standard name is gamma, not BW
        self.beta = np.zeros((1, num_features), dtype=np.float32)  # standard name is beta, not Bb

        self.dLdgamma = np.zeros((1, num_features), dtype=np.float32)
        self.dLdbeta = np.zeros((1, num_features), dtype=np.float32)

    def forward(self, Z):
        self.Z = Z
        self.N = None # TODO

        self.M = None # TODO
        self.V = None # TODO

        self.NZ = None # TODO
        self.BZ = None # TODO

        return self.BZ
    
    def backward(self, dLdBZ):
        self.dLdgamma = None # TODO
        self.dLdbeta = None # TODO

        dLdNZ = None # TODO

        dLdZ = None # TODO

        return dLdZ