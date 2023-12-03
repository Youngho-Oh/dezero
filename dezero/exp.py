from function import *
from square import *

class Exp(Function) :
    def forward(self, x: Variable) -> Variable:
        return np.exp(x.data)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    