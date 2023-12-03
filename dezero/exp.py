from dezero import function
from dezero import variable
import numpy as np

class Exp(function.Function) :
    def forward(self, x: variable.Variable) -> variable.Variable:
        return variable.Variable(np.exp(x.data))
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx