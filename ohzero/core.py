import numpy as np
import weakref
import contextlib
from ohzero import utils
from ohzero import cuda

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
    
    def reshape(self, *shape) :
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)) :
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self) :
        return transpose(self)
    
    def sum(self, axis=None, keepdims=False) :
        return sum(self, axis, keepdims)

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
    
    @property
    def T(self) :
        return transpose(self)

    def to_cpu(self):
        if self.data is not None:
            self.data = cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = cuda.as_cupy(self.data)

class Parameter(Variable) :
    pass

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

class Reshape(Function) :
    def __init__(self, shape) -> None:
        self.shape = shape
    
    def forward(self, x) :
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

class Transpose(Function) :
    def __init__(self, axes=None) :
        self.axes = axes

    def forward(self, x) :
        y = x.transpose(self.axes)
        return y
    
    def backward(self, gy) :
        if self.axes is None :
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

class Neg(Function) :
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

def neg(x) :
    return Neg()(x)

class Add(Function) :
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape :
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1) :
    return Add()(x0, x1)

class Sub(Function) :
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape :
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def sub(x0, x1) :
    return Sub()(x0, x1)

def rsub(x0, x1) :
    return Sub()(x1, x0)

class Mul(Function) :
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        # x0, x1 = self.inputs[0], self.inputs[1]
        # return gy * x1 , gy * x0
        gx0, gx1 = gy * self.inputs[1], gy * self.inputs[0]
        if self.x0_shape != self.x1_shape :
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        
        return gx0, gx1

def mul(x0, x1) :
    return Mul()(x0, x1)

class Div(Function) :
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        # x0, x1 = self.inputs[0], self.inputs[1]
        # gx0 = gy / x1
        # gx1 = gy * (-x0 / x1 ** 2)
        # return gx0 , gx1
        gx0, gx1 = gy / self.inputs[1], gy * (-self.inputs[0] / (self.inputs[1] ** 2))
        if self.x0_shape != self.x1_shape :
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        
        return gx0, gx1
    
def div(x0, x1) :
    return Div()(x0, x1)

def rdiv(x0, x1) :
    return Div()(x1, x0)

class Square(Function) :
    def forward(self, x) :
        return (x ** 2)

    def backward(self, gy):
        # x = self.input.data
        # x = self.inputs[0].data
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

def square(x:Variable) -> Variable :
# def square(x) :
    f = Square()
    return f(x)

class Exp(Function) :
    def forward(self, x: Variable) -> Variable:
    # def forward(self, x) :
        # return Variable(np.exp(x.data))
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y
    
    def backward(self, gy):
        # x = self.input.data
        # x = self.inputs[0]
        # gx = np.exp(x) * gy
        # return gx
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

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

def pow(x, c) :
    return Pow(c)(x)

class Sin(Function) :
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x) :
    return Sin()(x)

class Cos(Function) :
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x) :
    return Cos()(x)

class Tanh(Function) :
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx

def tanh(x) :
    return Tanh()(x)

class Sum(Function) :
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        
    def forward(self, x) :
        self.x_shape = x.shape
        # y = x.sum()
        # type of x is numpy.ndarray
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

# def sum(x) :
def sum(x, axis=None, keepdims=False) :
    # return Sum()(x)
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        # print("CALL BroadcastTo")
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        # print("CALL SumTo")
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W) :
        y = x.dot(W)

        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW

def matmul(x, W) :
    return MatMul()(x, W)

class MeanSquredError(Function) :
    def forward(self, x0, x1) :
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy) :
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0

        return gx0, gx1

def mean_squred_error(x0, x1) :
    return MeanSquredError()(x0, x1)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

def linear_simple(x, W, b=None) :
    t = matmul(x, W)
    if b is None :
        return t
    
    y = t + b
    t.data = None   # Release t.data (ndarray) for memory efficiency

    return y

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

def sigmoid_simple(x) :
    x = as_variable(x)
    y = 1 / (1 + exp(-x))

    return y

class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)

def as_variable(obj) :
    if isinstance(obj, Variable) :
        return obj
    elif not isinstance(obj, np.ndarray) :
        return Variable(np.array(obj))
    return Variable(obj)

def reshape(x, shape) :
    if x.shape == shape :
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x, axes=None) :
    return Transpose(axes)(x)

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
    Variable.__getitem__ = get_item

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

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

# =============================================================================
# softmax_simple, softmax, softmax_cross_entropy_simple, softmax_cross_entropy
# =============================================================================

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

# =============================================================================
# clip
# =============================================================================

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

# =============================================================================
# accuracy, as_array
# =============================================================================

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()

    return Variable(as_array(acc))

def as_array(x) :
    if np.isscalar(x) :
        return np.array(x)
    return x