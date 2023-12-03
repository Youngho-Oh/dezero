
class Variable :
    def __init__(self, data:float) -> None:
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func) -> None :
        self.creator = func
    
    def backward(self) -> None :
        f = self.creator
        if f is not None :
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()