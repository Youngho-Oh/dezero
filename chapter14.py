from ohzero import Variable
from ohzero.add import add
import numpy as np

x = Variable(np.array(3.0))
y = add(x, x)
# print('y', y.data)
y.backward()
print(x.grad)

x.cleargrad()
y = add(add(x,x),x)
y.backward()
print(x.grad)