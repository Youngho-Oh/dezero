if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ohzero
from ohzero.core import *
from ohzero.models import MLP
from ohzero import optimizers
import math
import numpy as np
import matplotlib.pyplot as plt
from ohzero import datasets

max_epoch = 300
batch_size = 30
hidden_size = 19
lr = 1.0

# x, t = ohzero.get_spiral(train=True)
train_set = datasets.Spiral()
model = MLP((hidden_size, hidden_size, hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch) :
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter) :
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' %(epoch + 1, avg_loss))

# x = np.array([data[0] for data in train_set])
# t = np.array([data[1] for data in train_set])

# # Plot boundary area the model predict
# h = 0.001
# x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
# y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# X = np.c_[xx.ravel(), yy.ravel()]

# with ohzero.no_grad():
#     score = model(X)
# predict_cls = np.argmax(score.data, axis=1)
# Z = predict_cls.reshape(xx.shape)
# plt.contourf(xx, yy, Z)

# # Plot data points of the dataset
# N, CLS_NUM = 100, 3
# markers = ['o', 'x', '^']
# colors = ['orange', 'blue', 'green']
# for i in range(len(x)):
#     c = t[i]
#     plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
# plt.show()