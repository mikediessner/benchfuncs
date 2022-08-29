import numpy as np
from benchfuncs import BenchmarkFunction
from typing import Optional


class Hartmann3D(BenchmarkFunction):

    def __init__(self, noise: Optional[float]=0.0, minimise: Optional[bool]=True) -> None:

        self.dims = 3
        self.bounds = np.array([[0.0, ] * self.dims, [1.0, ] * self.dims]).T
        self.optimum = {"inputs": np.array([[0.114614, 0.555649, 0.852547]]), 
                        "output": np.array([-3.86278])}
        self.noise = noise
        self.minimise = minimise

        self.a = np.array([1.0, 1.2, 3.0, 3.2]).T
        self.A = np.array([[3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0],
                           [3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0]])
        self.P = 10**-4 * np.array([[3689.0, 1170.0, 2673.0],
                                    [4699.0, 4387.0, 7470.0],
                                    [1091.0, 8732.0, 5547.0],
                                    [ 381.0, 5743.0, 8828.0]])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        
        # find number of points
        if len(x.shape) == 1:
            n = 1
            x = x.reshape((1, self.dims))
        else:
            n = x.shape[0]
        
        # create output array
        f = np.zeros(n)

        # iterate over points
        for i in range(n):

            # compute output
            y = -np.sum(self.a * np.exp(-np.sum(self.A * (x[i, :] - self.P)**2, axis=-1)), axis=-1)

            # turn into maximisation problem
            if not self.minimise:
                y = -y

            # add noise
            noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
            f[i] = y + noise

        return f


class Hartmann6D(BenchmarkFunction):

    def __init__(self, noise: Optional[float]=0.0, minimise: Optional[bool]=True) -> None:

        self.dims = 6
        self.bounds = np.array([[0.0, ] * self.dims, [1.0, ] * self.dims]).T
        self.optimum = {"inputs": np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]), 
                        "output": np.array([-3.32237])}
        self.noise = noise
        self.minimise = minimise

        self.a = np.array([1.0, 1.2, 3.0, 3.2]).T
        self.A = np.array([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
                           [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
                           [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
                           [17.0,  8.0, 0.05, 10.0,  0.1, 14.0]])
        self.P = 10**-4 * np.array([[1312.0, 1696.0, 5569.0,  124.0, 8283.0, 5886.0],
                                    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                                    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                                    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0,  381.0]])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        
        # find number of points
        if len(x.shape) == 1:
            n = 1
            x = x.reshape((1, self.dims))
        else:
            n = x.shape[0]
        
        # create output array
        f = np.zeros(n)

        # iterate over points
        for i in range(n):

            # compute output
            y = -np.sum(self.a * np.exp(-np.sum(self.A * (x[i, :] - self.P)**2, axis=-1)), axis=-1)

            # turn into maximisation problem
            if not self.minimise:
                y = -y

            # add noise
            noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
            f[i] = y + noise

        return f
