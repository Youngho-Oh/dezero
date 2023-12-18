import numpy as np

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