if '__file__' in globals() :
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import ohzero

max_epoch = 3
batch_size = 100

train_set = ohzero.datasets.MNIST(train=True)
train_loader = ohzero.DataLoader(train_set, batch_size)
model = ohzero.models.MLP((1000, 10))
optimizer = ohzero.optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch) :
    sum_loss, sum_accuracy = 0, 0

    for x, t in train_loader :
        y = model(x)
        loss = ohzero.core.softmax_cross_entropy(y, t)
        accuracy = ohzero.core.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_accuracy += float(accuracy.data) * len(t)
    
    print('epoch: {}, loss: {:.4f}, accuracy: {:.4f}'.format(epoch + 1, sum_loss / len(train_set), sum_accuracy / len(train_set)))

model.save_weights('my_mlp.npz')