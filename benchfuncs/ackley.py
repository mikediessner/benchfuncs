import numpy as np
from benchmark_function import BenchmarkFunction


class Ackley3D(BenchmarkFunction):

    def __init__(self, noise: float = 0.0, minimise: bool = True):

        self.dims = 3
        self.bounds = np.array([[-0.0, ] * self.dims, [1.0, ] * self.dims])
        self.optimum = {"inputs": np.array([[0.114614, 0.555649, 0.852547] * self.dims]), "ouput": np.array([[-3.86278]])}
        self.noise = noise
        self.minimise = minimise

        self.a = np.array([1.0, 1.2, 3.0, 3.2]).T
        self.A = np.array([[3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0],
                           [3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0]])
        self.P = 10**-4 * np.array([[3689.0, 1170.0, 2673.0],
                                    [4699.0, 4387.0, 7470.0],
                                    [1091.0, 8732.0, 5547.0],
                                    [ 381.0, 5743.0, 8828.0]])

    def __call__(self, x: np.array):
        
        # compute output
        y = -np.sum(self.a * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
