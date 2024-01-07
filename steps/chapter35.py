if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *
from ohzero.utils import plot_dot_graph
import matplotlib.pyplot as plt

x = Variable(np.array(1.0))
x.name = 'x'
y = tanh(x)
y.name = 'y'
y.backward(create_graph=True)

iters = 8

for i in range(iters) :
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
print(gx)
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=True, to_file='tanh.svg')