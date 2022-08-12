import numpy as np
from benchmark_function import BenchmarkFunction


class Levy(BenchmarkFunction):

    def __init__(self, dims: int, noise: float = 0.0, minimise: bool = True):

        self.dims = dims
        self.bounds = np.array([[-10., ] * dims, [10., ] * dims])
        self.optimum = {"inputs": np.array([[1.0, ] * dims]), "ouput": np.array([[0.0]])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.array):
        
        # compute output
        w = 1.0 + (x - 1.0)/4.0
        term_1 = np.sin(np.pi * w[:, 0])**2
        term_2 = np.sum(w[:, :-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0)**2)
        term_3 = (w[:, -1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1])**2)
        y = term_1 + term_2 + term_3

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
