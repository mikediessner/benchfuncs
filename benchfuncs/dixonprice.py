import numpy as np
from benchfuncs import BenchmarkFunction
from typing import Optional


class DixonPrice(BenchmarkFunction):

    def __init__(self, dims: int, noise: Optional[float]=0.0, minimise: Optional[bool]=True) -> None:

        ii = np.arange(1, dims+1)
        optimal_xs = 2.0**( -(2.0**ii - 2.0)/2.0**ii )

        self.dims = dims
        self.bounds = np.array([[-10.0, ] * dims, [10.0, ] * dims]).T
        self.optimum = {"inputs": optimal_xs.reshape((1, self.dims)), "output": np.array([0.0])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.ndarray) -> np.ndarray:

        # reshape if x is a single point
        if len(x.shape) == 1:
            x = x.reshape((1, self.dims)) 
        
        # compute output
        ii = np.arange(2, self.dims+1)
        term_1 = (x[:, 0] - 1.0)**2
        term_2 = np.sum(ii * (2.0*x[:, 1:]**2 - x[:, :-1])**2, axis=-1)
        y = term_1 + term_2

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
        f = y + noise

        return f
