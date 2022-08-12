import numpy as np
from benchmark_function import BenchmarkFunction


class Ackley(BenchmarkFunction):

    def __init__(self, dims: int, noise: float = 0.0, minimise: bool = True):

        self.dims = dims
        self.bounds = np.array([[-32.768, ] * dims, [32.768, ] * dims])
        self.optimum = {"inputs": np.array([[0.0, ] * dims]), "ouput": np.array([[0.0]])}
        self.noise = noise
        self.minimise = minimise

        self.a = 20.0
        self.b = 0.2
        self.c = 2.0*np.pi

    def __call__(self, x: np.array):
        
        # compute output
        term_1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2, axis=1)))
        term_2 = -np.exp(np.mean(np.cos(self.c * x), axis=1))
        y = term_1 + term_2 + self.a + np.exp(1.0)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
