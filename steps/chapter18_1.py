from ohzero import Variable
from ohzero.square import square
from ohzero.config import Config
import numpy as np

from memory_profiler import profile

import contextlib

# python3 -m memory_profiler chapter17.py

@profile
def call() :
    Config.enable_backprop = False
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))

    x = y = None

    Config.enable_backprop = True
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()

@profile
def call_2() :
    x = Variable(np.array(2.0))
    y = square(x)

@contextlib.contextmanager
def using_config(name, value) :
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield
    finally :
        setattr(Config, name, old_value)

def no_grad() :
    return using_config('enable_backprop', False)

# call()
with no_grad() :
    call_2()