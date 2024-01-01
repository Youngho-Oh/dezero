if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from ohzero import Variable
from ohzero.utils import plot_dot_graph

def rosenbrock(x0, x1) :
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
# iters = 1000
# iters = 50000
iters = 100000

for i in range(iters) :
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
    print(x0, x1)