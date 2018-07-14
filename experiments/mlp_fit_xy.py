from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import data as gdata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import mxnet


losser = gluon.loss.L2Loss()
X = nd.random_uniform(-100, 100, shape=(1000))
Y = nd.random_uniform(-100, 100, shape=(1000))
XY = nd.concat(X.reshape((-1, 1)), Y.reshape((-1, 1)), dim=1)
Z = nd.exp(- nd.abs(X - Y))


figure = plt.figure()
axes = plt.subplot(111, projection='3d')
axes.scatter(X.asnumpy(), Y.asnumpy(), Z.asnumpy(), 'r.')

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(200, activation='relu'))
net.add(gluon.nn.Dense(200, activation='relu'))
net.add(gluon.nn.Dense(1))
net.collect_params().initialize(mxnet.initializer.Xavier())

batch_size = 1000
dataset = gdata.ArrayDataset(XY, Z)
dataiter = gdata.DataLoader(dataset, batch_size=batch_size)
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'beta1': .9, 'beta2': .999})
# trainer = gluon.Trainer(net.collect_params(), 'sgd',
#                         {'learning_rate': .0000001})

for epoch in range(2000):
    train_loss = 0.
    train_acc = 0.
    for data, label in dataiter:
        with autograd.record():
            output = net(data)
            loss = losser(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
    print(train_loss)

Z = net(XY)
axes.scatter(X.asnumpy(), Y.asnumpy(), Z.asnumpy(), 'b.')
plt.show()
