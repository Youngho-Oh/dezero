from ohzero import Variable
from ohzero.square import square
import numpy as np

from memory_profiler import profile

# python3 -m memory_profiler chapter17.py

@profile
def call() :
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))
    print(x.data)
    print(y.data)

for i in range(10) :    
    call()