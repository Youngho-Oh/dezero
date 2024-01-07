import numpy as np
import weakref
import contextlib

class Config :
    enable_backprop = True

class Variable :
    def __init__(self, data:np.ndarray, name=None) -> None:
    # def __init__(self, data) :
        if data is not None :
            if isinstance(data, Variable) :
                data = data.data
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
    
    # def __mul__(self, other) :
    #     return mul(self, other)
    
    # def __add__(self, other) :
    #     return add(self, other)

    __array_priority__ = 200
    
    def set_creator(self, func) -> None :
    # def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False, create_graph=False) -> None :
    # def backward(self) :
        if self.grad is None :
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

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
            
            with using_config('enable_backprop', create_graph) :
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
        inputs = [as_variable(x) for x in inputs]
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

class Neg(Function) :
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Add(Function) :
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Sub(Function) :
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

class Mul(Function) :
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs[0], self.inputs[1]
        return gy * x1 , gy * x0

class Div(Function) :
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0 , gx1

class Square(Function) :
    def forward(self, x) :
        return (x ** 2)

    def backward(self, gy):
        # x = self.input.data
        # x = self.inputs[0].data
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

class Exp(Function) :
    def forward(self, x: Variable) -> Variable:
    # def forward(self, x) :
        return Variable(np.exp(x.data))
    
    def backward(self, gy):
        # x = self.input.data
        x = self.inputs[0]
        gx = np.exp(x) * gy
        return gx

class Pow(Function) :
    def __init__(self, c) :
        self.c = c

    def forward(self, x) :
        return x ** self.c
    
    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

class Sin(Function) :
    def forward(self, x) :
        y = np.sin(x)
        return y
    
    def backward(self, gy) :
        x, = self.inputs
        gx = gy * cos(x)
        return gx

class Cos(Function) :
    def forward(self, x) :
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Tanh(Function) :
    def forward(self, x) :
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def as_variable(obj) :
    if isinstance(obj, Variable) :
        return obj
    elif not isinstance(obj, np.ndarray) :
        return Variable(np.array(obj))
    return Variable(obj)

def neg(x) :
    return Neg()(x)

def add(x0, x1) :
    return Add()(x0, x1)

def sub(x0, x1) :
    return Sub()(x0, x1)

def rsub(x0, x1) :
    return Sub()(x1, x0)

def mul(x0, x1) :
    return Mul()(x0, x1)

def div(x0, x1) :
    return Div()(x0, x1)

def rdiv(x0, x1) :
    return Div()(x1, x0)

def sin(x) :
    return Sin()(x)

def cos(x) :
    return Cos()(x)

def tanh(x) :
    return Tanh()(x)

def square(x:Variable) -> Variable :
# def square(x) :
    f = Square()
    return f(x)

def exp(x:Variable) -> Variable :
# def exp(x) :
    f = Exp()
    return f(x)

def pow(x, c) :
    return Pow(c)(x)

def setup_variable() :
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow

@contextlib.contextmanager
def using_config(name, value) :
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield
    finally :
        setattr(Config, name, old_value)

def no_grad() :
    return using_config('enable_backprop', False)