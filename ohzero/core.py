from ohzero.config import Config
import numpy as np
import weakref

class Variable :
    def __init__(self, data:np.ndarray, name=None) -> None:
    # def __init__(self, data) :
        if data is not None :
            if not isinstance(data, np.ndarray) :
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def __len__(self) :
        return len(self.data)
    
    def __repr__(self) :
        if self.data is None :
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def __mul__(self, other) :
        return mul(self, other)
    
    def set_creator(self, func) -> None :
    # def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False) -> None :
    # def backward(self) :
        if self.grad is None :
            self.grad = np.ones_like(self.data)

        # funcs = [self.creator]
        funcs = []
        seen_set = set()

        def add_func(f) :
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs :
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple) :
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs) :
                if x.grad is None :
                    x.grad = gx
                else :
                    x.grad = x.grad + gx

                if x.creator is not None :
                    # funcs.append(x.creator)
                    add_func(x.creator)
            
            if not retain_grad :
                for output in f.outputs :
                    output().grad = None
    
    def cleargrad(self) :
        self.grad = None
    
    @property
    def shape(self) :
        return self.data.shape
    
    @property
    def ndim(self) :
        return self.data.ndim
    
    @property
    def size(self) :
        return self.data.size
    
    @property
    def dtype(self) :
        return self.data.dtype

class Function :
    # def __call__(self, input) :
    def __call__(self, *inputs) -> Variable :
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple) :
            ys = (ys, )
        outputs = [Variable(self.__as_array(y)) for y in ys]

        if Config.enable_backprop :
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

class Add(Function) :
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Mul(Function) :
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1 , gy * x0

class Square(Function) :
    def forward(self, x) :
        return (x ** 2)

    def backward(self, gy):
        # x = self.input.data
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function) :
    def forward(self, x: Variable) -> Variable:
    # def forward(self, x) :
        return Variable(np.exp(x.data))
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def add(x0, x1) :
    return Add()(x0, x1)

def mul(x0, x1) :
    return Mul()(x0, x1)

def square(x:Variable) -> Variable :
# def square(x) :
    f = Square()
    return f(x)

def exp(x:Variable) -> Variable :
# def exp(x) :
    f = Exp()
    return f(x)