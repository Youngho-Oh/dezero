from ohzero import Variable
from ohzero.square import square
from ohzero.add import add
import numpy as np

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)