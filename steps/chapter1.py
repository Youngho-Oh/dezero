# import config
from ohzero import Variable, Square, Exp
from ohzero.square import square
from ohzero.exp import exp
import numpy as np


# # A = Square()
# # B = Exp()
# # C = Square()

x = Variable(np.array(0.5))
y = Variable(1.0)
# # a = A(x)
# # b = B(a)
# # y = C(b)

# # assert y.creator == C
# # assert y.creator.input == b
# # assert y.creator.input.creator == B
# # assert y.creator.input.creator.input == a
# # assert y.creator.input.creator.input.creator == A
# # assert y.creator.input.creator.input.creator.input == x

# # y.grad = np.array(1.0)

# # C = y.creator
# # b = C.input
# # b.grad = C.backward(y.grad)
# # B = b.creator
# # a = B.input
# # a.grad = B.backward(b.grad)
# # A = a.creator
# # x = A.input
# # x.grad = A.backward(a.grad)
# # print(x.grad)
y = square(exp(square(x)))
y.backward()
print(x.grad)