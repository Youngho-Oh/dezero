from ohzero.function import Function
from ohzero.variable import Variable
import numpy as np

class Square(Function) :
    def forward(self, x) :
        return (x ** 2)

    def backward(self, gy):
        # x = self.input.data
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x:Variable) -> Variable :
# def square(x) :
    f = Square()
    return f(x)
    