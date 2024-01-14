if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = reshape(x, (6,))
# y.backward(retain_grad=True)
# print(x.grad)

# x = Variable(np.random.randn(1,2,3))
# print(x)
# y = x.reshape((2,3))
# print(y)
# y = x.reshape(3,2)
# print(y)


# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# # y = transpose(x)
# y = x.T
# print(y)
# y.backward()
# print(x.grad)

A, B, C, D = 1, 2, 3, 4
x = np.random.rand(A, B, C, D)
print(x)
print("==================")
y = x.transpose(1, 0, 3, 2)
print(y)