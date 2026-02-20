import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # Get the input dimension and the output width 
        self.A = A
        batch_size, in_channels, input_width = A.shape
        output_width = (input_width - 1) * self.upsampling_factor + 1

        # Output array
        Z = np.zeros((batch_size, in_channels, output_width))

        # Place the original values
        Z[:, :, ::self.upsampling_factor] = A

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = dLdZ[:, :, ::self.upsampling_factor]
        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A_shape = A.shape

        return A[:, :, ::self.downsampling_factor]

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """


        # Create zeros for the input arrays
        dLdA = np.zeros(self.A_shape)  # TODO

        # Place the gradients back
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        # Get the input dimension and the output width
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = (input_height - 1) * self.upsampling_factor + 1
        output_width  = (input_width  - 1) * self.upsampling_factor + 1

        # Create zeros for the output arrays
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        
        # Place thsi original values back at the positions
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.A_shape = A.shape

        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros(self.A_shape)
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
