from dezero.variable import Variable
import numpy as np

class Function :
    def __call__(self, input:Variable) -> Variable :
    # def __call__(self, input) :
        self.input = input
        self.output = Variable(self.__as_array(self.forward(self.input)))
        self.output.set_creator(self)
        return self.output
    
    def forward(self, x:Variable) -> Variable :
    # def forward(self, x) :
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()
    
    def __as_array(self, x) :
        if np.isscalar(x) :
            return np.array(x)
        return x