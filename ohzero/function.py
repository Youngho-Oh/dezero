from ohzero.variable import Variable
import numpy as np
import weakref

class Function :
    # def __call__(self, input) :
    def __call__(self, *inputs) -> Variable :
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple) :
            ys = (ys, )
        outputs = [Variable(self.__as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])

        for output in outputs :
            output.set_creator(self)

        self.inputs = inputs
        # self.outputs = outputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    # def forward(self, x) :
    def forward(self, x:Variable) -> Variable :
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()
    
    def __as_array(self, x) :
        if np.isscalar(x) :
            return np.array(x)
        return x