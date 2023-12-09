from dezero.function import Function
from dezero.variable import Variable
import numpy as np

class Exp(Function) :
    def forward(self, x: Variable) -> Variable:
    # def forward(self, x) :
        return Variable(np.exp(x.data))
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x:Variable) -> Variable :
# def exp(x) :
    f = Exp()
    return f(x)