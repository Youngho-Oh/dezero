if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *
from ohzero.utils import plot_dot_graph
import matplotlib.pyplot as plt

x = Variable(np.linspace(-7,7,200))
y = sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(4) :
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(labels) :
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()