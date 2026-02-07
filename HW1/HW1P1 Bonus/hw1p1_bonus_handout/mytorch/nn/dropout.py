# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # Generate binary mask
            self.mask = np.random.binomial(1, 1 - self.p, x.shape)

            # Mask to input and scale
            return x * self.mask / (1 - self.p)
            
        else:
            # No dropout, return inout unchaged
            return x
		
    def backward(self, delta):
        # TODO: Multiply mask with delta and return

        return delta * self.mask       