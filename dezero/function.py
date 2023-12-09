from dezero.variable import Variable

class Function :
    def __call__(self, input:Variable) -> Variable :
        self.input = input
        self.output = self.forward(self.input)
        self.output.set_creator(self)
        return self.output
    
    def forward(self, x:Variable) -> Variable :
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()
    