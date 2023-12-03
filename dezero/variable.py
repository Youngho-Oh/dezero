import numpy as np

class Variable :
    def __init__(self, data:float) -> None:
        self.data = data
        self.grad = None
        