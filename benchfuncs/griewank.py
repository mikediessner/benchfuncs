from pkgutil import iter_importers
import numpy as np
from benchmark_function import BenchmarkFunction


class Griewank(BenchmarkFunction):

    def __init__(self, dims: int, noise: float = 0.0, minimise: bool = True):

        self.dims = dims
        self.bounds = np.array([[-600.0, ] * dims, [600.0, ] * dims])
        self.optimum = {"inputs": np.array([[0.0, ] * dims]), "ouput": np.array([[0.0]])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.array):
        
        # compute output
        ii = np.arange(1, self.dims+1)
        y = np.sum(x**2/4000.0, axis=1) - np.prod(np.cos(x / np.sqrt(ii)), axis=1) + 1

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
