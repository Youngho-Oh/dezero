if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *

# x = Variable(np.array([1, 2, 3, 4, 5, 6]))
# x = Variable(np.array([[1,2,3],[4,5,6]]))
# y = sum(x)
# y.backward()

# print(y)
# print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)