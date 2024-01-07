if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *
from ohzero.utils import plot_dot_graph
import matplotlib.pyplot as plt

def f(x) -> Variable :
    y = x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

# example about double backprop
z = gx ** 3 + y
z.backward()
print(x.grad)