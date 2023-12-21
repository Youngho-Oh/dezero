from ohzero.core import Variable
from ohzero.core import add
# from ohzero.mul import mul
import numpy as np

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a, b), c)
temp = a * b
print(temp)
y = add(temp, c)

y.backward()

print(y)
print(a.grad)
print(b.grad)