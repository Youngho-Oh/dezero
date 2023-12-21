from ohzero.function import Function
# from ohzero.variable import Variable

class Mul(Function) :
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1 , gy * x0

def mul(x0, x1) :
    return Mul()(x0, x1)