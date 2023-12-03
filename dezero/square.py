from dezero import function
from dezero import variable

class Square(function.Function) :
    def forward(self, x: variable.Variable) -> variable.Variable:
        return variable.Variable(x.data ** 2)

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    