import numpy as np
import scipy


### No need to modify Identity class
class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.
    """

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dLdA):
        dAdZ = self.A - self.A * self.A

        # Chain rule here
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    Tanh activation function.
    """

    def forward(self, Z):
        eZ = np.exp(Z)
        eNegZ = np.exp(-Z)
        self.A = (eZ - eNegZ) / (eZ + eNegZ)
        return self.A
    
    def backward(self, dLdA):
        dAdZ = 1 - self.A * self.A

        # Chain rule here
        dLdZ = dLdA * dAdZ
        return dLdZ
    

class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    def forward(self, Z):
        self.Z = Z
        self.A = np.maximum(0, Z)
        return self.A
    
    def backward(self, dLdA):
        dAdZ = (self.Z > 0).astype("f")
        dLdZ = dLdA * dAdZ
        return dLdZ



class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    """
    def forward(self, Z):
        self.Z = Z
        self.Phi = 0.5 * (1.0 + scipy.special.erf(Z / np.sqrt(2.0)))
        self.A = Z * self.Phi
        return self.A
    
    def backward(self, dLdA):
        # standard normal pdf
        phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-(self.Z ** 2) / 2.0)
        dAdZ = self.Phi + self.Z * phi
        dLdZ = dLdA * dAdZ
        return dLdZ

class Swish:
    """
    Swish activation function.
    """
    def __init__(self, beta=1.0):
        self.beta = np.array([beta], dtype="f")
        self.dLdbeta = np.zeros((1,), dtype="f")

    def forward(self, Z):
        self.Z = Z
        self.sig = 1 / (1 + np.exp(-(self.beta[0] * Z)))
        self.A = Z * self.sig
        return self.A
    
    def backward(self, dLdA):
        s = self.sig
        dAdZ = s + self.beta[0] * self.Z * s * (1-s)
        dLdZ = dLdA * dAdZ

        dAdbeta = (self.Z ** 2) * s * (1 - s)
        self.dLdbeta = np.array([np.sum(dLdA * dAdbeta)], dtype="f")
        return dLdZ

class Softmax:
    """
    Softmax activation function.

    ToDO:
    On same lines as above, create your own mytorch.nn.Softmax!
    Complete the 'forward' function.
    Complete the 'backward' function.
    Read the writeup (Hint: Softmax Section) for further details on Softmax forward and backward expressions.
    Hint: You read more about `axis` and `keep_dims` attributes, helpful for future homeworks too.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        Note: How can we handle large overflow values? Hint: Check numerical stability.
        """
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_shift)
        self.A = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.A  # TODO - What should be the return value?

    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N = dLdA.shape[0]
        C = dLdA.shape[1]  # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C), dtype="f")

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            # Hint: Jacobian matrix for softmax is a _×_ matrix, but what is _ here?
            J = np.zeros((C, C), dtype="f")

            # Fill the Jacobian matrix, please read the writeup for the conditions.
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1.0 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
            # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
            dLdZ[i, :] = dLdA[i, :] @ J  

        return dLdZ  # TODO - What should be the return value?
