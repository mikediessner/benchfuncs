import numpy as np
from benchfuncs import BenchmarkFunction
from typing import Optional


class Zakharov(BenchmarkFunction):

    def __init__(self, dims: int, noise: Optional[float]=0.0, minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = np.array([[-5.0, ] * dims, [10.0, ] * dims]).T
        self.optimum = {"inputs": np.array([[0.0, ] * dims]), "output": np.array([0.0])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.ndarray) -> np.ndarray:
        
        # compute output
        ii = np.arange(1, self.dims+1)
        term_1 = np.sum(x**2, axis=-1)
        term_2 = np.sum(0.5 * ii * x, axis=-1) 
        y = term_1 + term_2**2 + term_2**4

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
        f = y + noise

        return f
