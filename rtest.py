from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

# Import the DEoptim package
DEoptim = importr("DEoptim")

# The objective function (Rosenbrock function)
def Rosenbrock(x):
    x = FloatVector(x)
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Lower and upper bounds
lower = [-10, -10]
upper = [10, 10]

# DEoptim.control parameters
control = {'NP': 80, 'itermax': 400, 'F': 1.2, 'CR': 0.7}

# Run DEoptim
result = DEoptim.DEoptim(DEoptim_fn=Rosenbrock, lower=FloatVector(lower), upper=FloatVector(upper), control=control)
print(result.rx2('optim'))
