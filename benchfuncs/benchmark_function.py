import numpy as np
import torch


class BenchmarkFunction:

    def __init__(self):
        pass

    def torch(self, x: np.array):
        """
        Wrapper function to use in torch environment.
        """

        x = x.numpy()
        y = self.__call__(x)
        y = torch.from_numpy(y)

        return y
