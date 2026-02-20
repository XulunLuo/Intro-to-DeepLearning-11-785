import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape
        kernel = self.kernel

        # Output dimensions
        output_width = input_width - kernel + 1
        output_height = input_height - kernel + 1

        self.A = A

        # Initialize output
        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        # Store max indicies for backward pass
        self.max_indices = np.zeros((batch_size, in_channels, output_width, output_height, 2), dtype=int)

        # Let's max pooling for now
        for a in range (batch_size):
            for b in range (in_channels):
                for c in range (output_width):
                    for d in range (output_height):
                        window = A[a, b, c:c+kernel, d:d+kernel]

                        # Got the masx value out of it
                        Z[a, b, c, d] = np.max(window)

                        # Store position of max
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[a, b, c, d] = [max_pos[0], max_pos[1]]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, input_width, input_height = self.A.shape
        kernel = self.kernel
        output_width, output_height = dLdZ.shape[2], dLdZ.shape[3]

        # Gradient
        dLdA = np.zeros(self.A.shape)

        # Backpropagate gradient only to max positions
        for a in range(batch_size):
            for b in range(in_channels):
                for c in range(output_width):
                    for d in range(output_height):
                        # Positions where all the max was found
                        max_c, max_d = self.max_indices[a, b, c, d]

                        # Match gradients to those positions
                        dLdA[a, b, c + max_c, d + max_d] += dLdZ[a, b, c, d]
        
        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, input_cahnnels, input_width, input_height = A.shape
        kernel = self.kernel

        output_width = input_width - kernel + 1
        output_height = input_height - kernel + 1

        self.input_shape = A.shape

        Z = np.zeros((batch_size, input_cahnnels, output_width, output_height))

        # Mean pooling
        for a in range(batch_size):
            for b in range(input_cahnnels):
                for c in range(output_width):
                    for d in range(output_height):
                        window = A[a, b, c:c+kernel, d:d+kernel]
                        Z[a, b, c, d]  = np.mean(window)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, input_channels, input_width, input_height = self.input_shape
        kernel = self.kernel
        output_width, output_height = dLdZ.shape[2], dLdZ.shape[3]

        # Initialize gradient
        dLdA = np.zeros(self.input_shape)

        # Gradient distribution
        for a in range(batch_size):
            for b in range(input_channels):
                for c in range(output_width):
                    for d in range(output_height):
                        gradients = dLdZ[a, b, c, d] / (kernel * kernel)
                        dLdA[a, b, c:c+kernel, d:d+kernel] += gradients

        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Apply stride-1 max pooling
        Z_stride1 = self.maxpool2d_stride1.forward(A)

        # Downsample
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Backprop through downsample
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)

        # Backprop through stride-1 max pool
        dLdA = self.maxpool2d_stride1.backward(dLdZ_stride1)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
         # Apply stride-1 mean pooling
        Z_stride1 = self.meanpool2d_stride1.forward(A)

        # Downsample
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Backprop through downsample
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)
        
        # Backprop through stride-1 mean pool
        dLdA = self.meanpool2d_stride1.backward(dLdZ_stride1)
        
        return dLdA
