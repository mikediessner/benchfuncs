import numpy as np
from benchmark_function import BenchmarkFunction


class DixonPrice(BenchmarkFunction):

    def __init__(self, dims: int, noise: float = 0.0, minimise: bool = True):

        ii = np.arange(1, dims+1)
        optimal_xs = 2.0**( -(2.0**ii - 2.0)/2.0**ii )

        self.dims = dims
        self.bounds = np.array([[-10.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": np.array([optimal_xs]), "ouput": np.array([[0.0]])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.array):
        
        # compute output
        ii = np.range(2, self.dims+1)
        term_1 = (x[:, 0] - 1.0)**2
        term_2 = np.sum(ii * (2.0*x[:, 1:]**2 - x[:, :-1])**2, axis=1)
        y = term_1 + term_2

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
