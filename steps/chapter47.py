if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *
import ohzero.layers as L
from ohzero.models import MLP
from ohzero import optimizers
from ohzero.functions import softmax_cross_entropy

# def softmax1d(x) :
#     x = as_variable(x)
#     y = exp(x)
#     sum_y = sum(y)
#     return y / sum_y

np.random.seed(0)

# # x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# # print(x)

# # y = x[1] #get_item(x, 1)
# # print(y)

# # y = x[:,2]
# # print(y)

model = MLP((10, 3))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])  # select solution in 4th x cases
y = model(x)
print(y)
loss = softmax_cross_entropy(y, t)
loss.backward()
print(loss)