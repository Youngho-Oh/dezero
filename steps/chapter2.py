from ohzero import Variable, Add
from ohzero.add import add
from ohzero.square import square
import numpy as np

xs = [Variable(np.array(2)), Variable(np.array(3))]
# f = Add()
# ys = f(xs)
# y = ys[0]
# y = f(xs[0], xs[1])
y = add(xs[0], xs[1])
print(y.data)

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)