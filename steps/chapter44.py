if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ohzero.core import *
import ohzero.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

model = L.Layer()
model.l1 = L.Linear(10)
model.l2 = L.Linear(1)

def predict(x) :
    y = model.l1(x)
    y = sigmoid(y)
    y = model.l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters) :
    y_pred = predict(x)
    loss = mean_squred_error(y, y_pred)
    # l1.cleargrads()
    # l2.cleargrads()
    model.cleargrads()
    loss.backward()

    for p in model.params() :
        p.data -= lr * p.grad.data
    
    if i % 1000 == 0 :
        print(loss)