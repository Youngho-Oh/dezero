if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *
from ohzero.utils import plot_dot_graph
import matplotlib.pyplot as plt

# Linear regression using mean squared error

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

x, y  = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x) :
    y = matmul(x, W) + b
    return y

# def mean_squred_error(x0, x1) :
#     diff = x0 - x1
#     return sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100
loss = None

for i in range(iters) :
    y_pred = predict(x)
    loss = mean_squred_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward(create_graph=True)

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)
    if i == 0 :
        plot_dot_graph(x.grad, verbose=False, to_file='mse.svg')