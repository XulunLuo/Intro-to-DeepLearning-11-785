import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape

        # Calculate output dimensions
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # Slide over height and width
        for h in range(output_height):
            for w in range(output_width): 
                window = A[:, :, h:h+self.kernel_size, w:w+self.kernel_size]
        
                # For each output channel
                for output in range(self.out_channels):
                    # Multiply and sum over in_channels, kernal_h, and kernel_w
                    Z[:, output, h, w] = np.sum(window * self.W[output], axis=(1, 2, 3))

                Z[:, :, h, w] += self.b

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        _, in_channels, input_height, input_width = self.A.shape

        # Intialize gradients
        self.dLdW = np.zeros(self.W.shape) # TODO
        self.dLdb = np.zeros(self.b.shape) # TODO
        dLdA = np.zeros(self.A.shape)

        # Compute dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # Compute dLdW and dLdA
        for h in range(output_height):
            for w in range(output_width):
                window = self.A[:, :, h:h+self.kernel_size, w:w+self.kernel_size]

                # Gradient
                gradient = dLdZ[:, :, h, w]

                # Compute dLdW
                for output in range(out_channels):
                    grad_boardcast = gradient[:, output].reshape(batch_size, 1, 1, 1)
                    self.dLdW[output] += np.sum(grad_boardcast * window, axis=0)

                # Compute dLdA
                for output in range(out_channels):
                    grad_boardcast = gradient[:, output].reshape(batch_size, 1, 1, 1)
                    temp_grad = grad_boardcast * self.W[output]
                    dLdA[:, :, h:h+self.kernel_size, w:w+self.kernel_size] += temp_grad # TODO

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)   # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        if self.pad > 0:
            A_padded = np.pad(A, pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=0)
        else:
            A_padded = A

        # Call Conv2d_stride1
        Z_stride1 = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z_stride1)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        dLdZ_stride1 = self.downsample2d.backward(dLdZ)
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_stride1)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]  # TODO
        else:
            dLdA = dLdA_padded

        return dLdA
