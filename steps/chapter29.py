if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero import Variable

def f(x) :
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x) :
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
y = 0
lr = 0.01
iters = 1000

print("<Start gradient descent Method>")
for i in range(iters) :
    y = f(x)
    
    x.cleargrad()
    y.backward()

    x.data -= lr * x.grad
    print(i, x)

print("<Start Newton Method>")
x = Variable(np.array(2.0))
y = 0
iters = 10

for i in range(iters) :
    y = f(x)

    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
    print(i, x)
