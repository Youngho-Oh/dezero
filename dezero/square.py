from dezero.function import Function
from dezero.variable import Variable
import numpy as np

class Square(Function) :
    def forward(self, x: Variable) -> Variable:
        return (x.data ** 2)

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x:Variable) -> Variable :
# def square(x) :
    f = Square()
    return f(x)
    