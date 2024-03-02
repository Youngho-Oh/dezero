if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ohzero
import ohzero.core as F
from ohzero.models import MLP
from ohzero import optimizers
import math
import numpy as np
import matplotlib.pyplot as plt
from ohzero import DataLoader
from ohzero import Spiral

lr = 1.0
hidden_size = 10
batch_size = 30
max_epoch = 300

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

epoch_label = []
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(max_epoch) :
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    train_loss.append(sum_loss / len(train_set))
    train_accuracy.append(sum_acc / len(train_set))

    sum_loss, sum_acc = 0, 0
    with ohzero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))

    test_loss.append(sum_loss / len(test_set))
    test_accuracy.append(sum_acc / len(test_set))
    epoch_label.append(epoch)

fig = plt.figure()
fig.set_facecolor('white')
ax = fig.add_subplot()

ax.plot(epoch_label, train_loss, label="train")
ax.plot(epoch_label, test_loss, label="test")

ax.legend()
plt.show()

fig = plt.figure()
fig.set_facecolor('white')
ax = fig.add_subplot()

ax.plot(epoch_label, train_accuracy, label="train")
ax.plot(epoch_label, test_accuracy, label="test")

ax.legend()
plt.show()

# y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
# t = np.array([1, 2, 0])
# acc = F.accuracy(y, t)
# print(acc)