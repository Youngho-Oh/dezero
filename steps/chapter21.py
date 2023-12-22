from ohzero.core import Variable
# from ohzero.mul import mul
import numpy as np

# x = Variable(np.array(2.0))

# # y = x + np.array(3.0)
# # y = x + 3.0
# y = 3.0 * x + 1.0

x = Variable(np.array([1.0]))
y = np.array([2.0]) + x
print(y)
y = x + np.array([2.0])
print(y)