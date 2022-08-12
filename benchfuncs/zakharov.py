import numpy as np
from benchmark_function import BenchmarkFunction


class Zakharov(BenchmarkFunction):

    def __init__(self, dims: int, noise: float = 0.0, minimise: bool = True):

        self.dims = dims
        self.bounds = np.array([[-5.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": np.array([[0.0, ] * dims]), "ouput": np.array([[0.0]])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.array):
        
        # compute output
        ii = np.arange(1, self.dims+1)
        term_1 = np.sum(x**2, axis=1)
        term_2 = np.sum(0.5 * ii * x, axis=1) 
        y = term_1 + term_2**2 + term_2**4

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
