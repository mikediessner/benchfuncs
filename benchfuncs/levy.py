import numpy as np
from benchfuncs import BenchmarkFunction
from typing import Optional


class Levy(BenchmarkFunction):

    def __init__(self, dims: int, noise: Optional[float]=0.0, minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = np.array([[-10., ] * dims, [10., ] * dims]).T
        self.optimum = {"inputs": np.array([[1.0, ] * dims]), "output": np.array([0.0])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.ndarray) -> np.ndarray:

        # reshape if x is a single point
        if len(x.shape) == 1:
            x = x.reshape((1, self.dims)) 
        
        # compute output
        w = 1.0 + (x - 1.0)/4.0
        term_1 = np.sin(np.pi * w[:, 0])**2
        term_2 = np.sum((w[:, :-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0)**2), axis=-1)
        term_3 = (w[:, -1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1])**2)
        y = term_1 + term_2 + term_3

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
        f = y + noise

        return f
