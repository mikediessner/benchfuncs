import numpy as np
from benchmark_function import BenchmarkFunction


class Hartmann3D(BenchmarkFunction):

    def __init__(self, noise: float = 0.0, minimise: bool = True):

        self.dims = 3
        self.bounds = np.array([[-0.0, ] * self.dims, [1.0, ] * self.dims])
        self.optimum = {"inputs": np.array([[0.114614, 0.555649, 0.852547] * self.dims]), 
                        "ouput": np.array([[-3.86278]])}
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

    def __call__(self, x: np.array):
        
        # compute output
        y = -np.sum(self.a * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))


class Hartmann6D(BenchmarkFunction):

    def __init__(self, noise: float = 0.0, minimise: bool = True):

        self.dims = 6
        self.bounds = np.array([[-0.0, ] * self.dims, [1.0, ] * self.dims])
        self.optimum = {"inputs": np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573] * self.dims]), 
                        "ouput": np.array([[-3.32237]])}
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

    def __call__(self, x: np.array):
        
        # compute output
        y = -np.sum(self.a * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = np.random.normal(loc=0, scale=noise, size=y.shape)
        f = y + noise

        return f.reshape((-1, 1))
