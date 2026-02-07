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
        self.N = Z.shape[0] # TODO

        # Mean Per Sample
        self.M = np.mean(Z, axis=1, keepdims=True) # TODO

        # Variance Per Sample
        self.V = np.var(Z, axis=1, keepdims=True) # TODO

        # Normalize
        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps) # TODO

        # Scale the thing and shift
        self.BZ = self.gamma * self.NZ + self.beta # TODO

        return self.BZ
    
    def backward(self, dLdBZ):

        # Gradient w.r.t gamma
        self.dLdgamma = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True) # TODO

        # Gradient w.r.t. beta
        self.dLdbeta = np.sum(dLdBZ, axis=0, keepdims=True) # TODO

        # Gradient w.r.t. normalized input
        dLdNZ = np.sum(dLdNZ, axis=1, keepdims=True) # TODO

        # Gradient w.r.t. input Z
        C = self.num_features

        # Sum of dLdNZ over features for each sample 
        sum_dLdNZ = np.sum(dLdNZ, axis=1, keepdims=True)

        # Sum of (dLdNZ * NZ) over features for each sample
        sum_dLdNZ_NZ = np.sum(dLdNZ * self.NZ, axis=1, keepdims=True)

        # Now apply the gradient formula
        dLdZ = (1.0 / C) / np.sqrt(self.V + self.eps) * (C * dLdNZ - sum_dLdNZ - self.NZ * sum_dLdNZ_NZ)# TODO

        return dLdZ