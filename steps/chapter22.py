from ohzero.core import Variable
# from ohzero.mul import mul
import numpy as np

# x = Variable(np.array(2.0))

# # y = x + np.array(3.0)
# # y = x + 3.0
# y = 3.0 * x + 1.0

x = Variable(np.array(2.0))
# y = -x
# print(y)
# print(x.grad)

y1 = 1.0 - x
print(y1)
y2 = x - 1.0
print(y2)

y = x ** 3
print(y)