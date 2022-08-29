import numpy as np
from benchfuncs import BenchmarkFunction
from typing import Optional


class Schwefel(BenchmarkFunction):

    def __init__(self, dims: int, noise: Optional[float]=0.0, minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = np.array([[-500.0, ] * dims, [500.0, ] * dims]).T
        self.optimum = {"inputs": np.array([[420.9687, ] * dims]), "output": np.array([0.0])}
        self.noise = noise
        self.minimise = minimise

    def __call__(self, x: np.ndarray) -> np.ndarray:
        
        # compute output
        y = 418.9829 * self.dims - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
        f = y + noise

        return f
