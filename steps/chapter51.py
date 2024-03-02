if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import ohzero
import matplotlib.pyplot as plt 

def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x

train_set = ohzero.datasets.MNIST(train=True, transform=f)
test_set = ohzero.datasets.MNIST(train=False, transform=f)

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_loader = ohzero.DataLoader(train_set, batch_size)
test_loader = ohzero.DataLoader(test_set, batch_size, shuffle=False)

# model = ohzero.models.MLP((hidden_size, 10))
model = ohzero.models.MLP((hidden_size, hidden_size, 10), activation=ohzero.core.relu)
# optimizer = ohzero.optimizers.SGD().setup(model)
optimizer = ohzero.optimizers.Adam().setup(model)

epoch_label = []
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader :
        y = model(x)
        loss = ohzero.core.softmax_cross_entropy(y, t)
        acc = ohzero.core.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    print('epoch : {}'.format(epoch + 1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    train_loss.append(sum_loss / len(train_set))
    train_accuracy.append(sum_acc / len(train_set))

    sum_loss, sum_acc = 0, 0
    with ohzero.no_grad():
        for x, t in test_loader :
            y = model(x)
            loss = ohzero.core.softmax_cross_entropy(y, t)
            acc = ohzero.core.accuracy(y, t)
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