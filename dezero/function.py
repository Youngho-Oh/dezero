from variable import *
from variable import Variable

class Function :
    def __call__(self, input:Variable) -> Variable :
        self.input = input
        return Variable(self.forward(input))
    
    def forward(self, x:Variable) -> Variable :
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()
    