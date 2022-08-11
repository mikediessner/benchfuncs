# IN DEVELOPMENT: benchfuncs

The `benchfuncs` package offers a selection of benchmark functions that can be used to test optimisation algorithms. It is implemented in numpy but a torch wrapper allows convenient use with torch optimisers as well.

```python
from benchfuncs import Sphere
from simplelhs import LatinHypercubeSampling
from scipy import minimize


# define benchmark function
noisy_sphere = Sphere(dims=2, noise=0.01)

# sample training data via a Latin Hypercube
lhs = LatinHypercubeSampling(noisy_sphere.dims)
x = lhs.random(10)

# get training data outpus
y = noisy_sphere(x)

# print results
print("Inputs: ", x)
print("Outputs: ", y)

# optimise benchmark function
x0 = np.random.uniform()
results = minimize(noisy_sphere, x0, method='L-BFGS-B')

# compare solution with global minimum
print("Solution: ", results["x"], results["fun"])
print("Optimum: ", noisy_sphere.optimum["inputs"], noisy_sphere.optimum["output"])
```
