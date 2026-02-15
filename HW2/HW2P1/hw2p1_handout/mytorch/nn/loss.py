import numpy as np

# Copy your Linear class from HW1P1 here
class MSELoss:
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        se = (A-Y) ** 2  # TODO
        sse = np.sum(se)  # TODO
        mse = sse / (N * C)

        return mse

    def backward(self):
        N = self.self.A
        C = self.self.Y

        # Calculate the gradient
        dLdA = (2 / (N *C)) * (self.A - self.Y)

        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        Ones_C = np.ones((C, 1), dtype="f")
        Ones_N = np.ones((N, 1), dtype="f")

        # Compute the exp of A first
        exp_A = np.exp(A)

        # Sum across the classes
        sum_exp_A = np.sum(exp_A, axis=1, keepdims=True)

        self.softmax = exp_A / sum_exp_A  # TODO
        crossentropy = -Y * np.log(self.softmax)  # TODO
        sum_crossentropy = np.sum(crossentropy)  # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):

        N = self.A.shape[0]

        dLdA = (self.softmax - self.Y) / N  # TODO

        return dLdA