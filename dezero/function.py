from dezero import variable

class Function :
    def __call__(self, input:variable.Variable) -> variable.Variable :
        self.input = input
        self.output = self.forward(self.input)
        self.output.set_creator(self)
        return self.output
    
    def forward(self, x:variable.Variable) -> variable.Variable :
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()
    