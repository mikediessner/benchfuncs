import numpy as np
from benchmark_function import BenchmarkFunction


class Rastrigin(BenchmarkFunction):

    def __init__(self, dims: int, noise: float = 0.0, minimise: bool = True):

        self.dims = dims
        self.bounds = np.array([[-5.12, ] * dims, [5.12, ] * dims])
        self.optimum = {"inputs": np.array([[0.0, ] * dims]), "ouput": np.array([[0.0]])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.array):
        
        # compute output
        y = 10*self.dims + np.sum(x**2 - 10*np.cos(2*np.pi*x))

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
