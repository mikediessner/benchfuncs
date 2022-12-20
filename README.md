# IN DEVELOPMENT: benchfuncs

The `benchfuncs` package offers a selection of benchmark functions that can be used to test optimisation algorithms. Below is an example that shows (a) how `bencfuncs` can be used to generate training points for modelling and machine learning and (b) how functions can be solved with an optimiser (in this case the L-BFGS-B solver from the `scipy.optimize`package.)

```python
from benchfuncs import Sphere
import numpy as np
from scipy.optimize import minimize


# define benchmark function
func = Sphere(dims=4)

# sample training data
x = np.random.uniform(low=func.bounds[:, 0], high=func.bounds[:, 1], size=(3, func.dims))

# get training data outpus
y = func(x)

# print results
print("Inputs: ", x)
print("Outputs: ", y)
```

```
Inputs:  [[-2.69264719  2.34062409 -0.16280928 -0.55581425]
          [ 1.60939843  0.28119955 -0.47671281 -0.51286048]
          [ 2.32721282  0.32551319 -1.25113608 -4.29779957]]
Outputs:  [13.0643064   3.15951746 25.55830099]
```

```python
# optimise benchmark function
x0 = np.random.uniform(low=func.bounds[:, 0], high=func.bounds[:, 1], size=func.dims)
results = minimize(func, x0, method='L-BFGS-B', bounds=func.bounds)

# compare solution with global minimum
print(f"Solution: \t Inputs: {results['x']} \t Output: {results['fun']}")
print(f"Optimum: \t Inputs: {func.optimum['inputs']} \t Output: {func.optimum['output']}")
```

```
Solution:        Inputs: [-4.92781912e-09 -4.96113870e-09 -4.99729987e-09 -5.03493624e-09]      Output: 9.921988747713389e-17
Optimum:         Inputs: [[0. 0. 0. 0.]]         Output: [0.]
```
