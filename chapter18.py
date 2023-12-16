from ohzero import Variable
from ohzero.add import add
import numpy as np

from memory_profiler import profile

# python3 -m memory_profiler chapter17.py

@profile
def call() :
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print(y.grad, t.grad)
    print(x0.grad, x1.grad)


call()