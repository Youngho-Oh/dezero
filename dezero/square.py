from function import *

class Square(Function) :
    def forward(self, x: Variable) -> Variable:
        return x.data ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    