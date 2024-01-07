if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero import Variable
from ohzero.utils import plot_dot_graph

def f(x) -> Variable :
    y = x ** 4 - 2 * x ** 2
    return y

# def gx2(x) :
    # return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters) :
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data