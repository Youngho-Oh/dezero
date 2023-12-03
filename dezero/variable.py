import numpy as np

class Variable :
    def __init__(self, data:float) -> None:
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func) -> None :
        self.creator = func