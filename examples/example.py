from benchfuncs import Sphere
import numpy as np
from scipy.optimize import minimize


# define benchmark function
func = Sphere(dims=4)

# sample training data for modelling
x = np.random.uniform(low=func.bounds[:, 0], high=func.bounds[:, 1], size=(3, func.dims))

# get training data outpus
y = func(x)

# print results
print("Inputs: ", x)
print("Outputs: ", y)

# optimise benchmark function
x0 = np.random.uniform(low=func.bounds[:, 0], high=func.bounds[:, 1], size=func.dims)
results = minimize(func, x0, method='L-BFGS-B', bounds=func.bounds)

# compare solution with global minimum
print(f"Solution: \t Inputs: {results['x']}, \t Output: {results['fun']}")
print(f"Optimum: \t Inputs: {func.optimum['inputs']} \t Output: {func.optimum['output']}")
