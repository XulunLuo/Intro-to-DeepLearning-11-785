# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_width = A.shape
        output_width = input_width - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_width))

        # Slide kernel across input
        for i in range(output_width):
            # Extract the input window
            window = A[:, :, i:i+self.kernel_size]

            for out in range(self.out_channels):
                # Multiply and sum 
                Z[:, out, i] = np.sum(window * self.W[out], axis=(1, 2))

            # Add bias
            Z[:, :, i] += self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_width = dLdZ.shape
        _, in_channels, input_width = self.A.shape

        # Initialize gradients
        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)  # TODO
        dLdA = np.zeros(self.A.shape)  # TODO

        # Compute dLdb
        self.dLdb = np.sum(dLdZ, axis=(0,2))

        # Compute dLdW and dLdA
        for i in range(output_width):
            # Extract the input window
            window = self.A[:, :, i:i+self.kernel_size] 

            # Gradient
            gradient = dLdZ[:, :, i]

            # Compute dLdW
            for out in range(out_channels):
                grad_boardcast = gradient[:, out].reshape(batch_size, 1, 1)
                
                # Multiply and sum 
                self.dLdW[out] += np.sum(grad_boardcast * window, axis=0)

            # Compute dLdA
            for out in range(out_channels):
                grad_boardcast = gradient[:, out].reshape(batch_size, 1, 1)

                # Multiply and accumulate
                dLdA[:, :, i:i+self.kernel_size] += grad_boardcast * self.W[out]

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input appropriately using np.pad() function
        if self.pad > 0:
            A_padded = np.pad(A, pad_width=((0, 0), (0, 0), (self.pad, self.pad)),
                         mode='constant', 
                         constant_values=0)

        # Call Conv1d_stride1
        else:
            A_padded = A

        # Apply stride-1 convolution
        Z_stride1 = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(Z_stride1)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # CCall Conv1d_stride1 backward
        dLdZ_stride1 = self.downsample1d.backward(dLdZ)

        # Backprop 
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_stride1)

        # Remove padding from gradient
        if self.pad > 0:
            # Keeps only the middle, removing pad from both sides
            dLdA = dLdA_padded[:, :, self.pad:-self.pad]
        else:
            # Unpad the gradient
            dLdA = dLdA_padded

        return dLdA
