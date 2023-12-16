import numpy as np

class Variable :
    def __init__(self, data:np.ndarray) -> None:
    # def __init__(self, data) :
        if data is not None :
            if not isinstance(data, np.ndarray) :
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func) -> None :
    # def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self) -> None :
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
    
    def cleargrad(self) :
        self.grad = None