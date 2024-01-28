if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)