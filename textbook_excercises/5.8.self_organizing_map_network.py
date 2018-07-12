import mxnet
import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet.gluon import data as gdata
from math import sqrt
import pandas as pd


class Som(nn.Block):
    def __init__(self, input_dim, num_center, **args):
        super(Som, self).__init__(**args)
        self.W = self.params.get('W', shape=(num_center, input_dim))

    def unify(self):
        params = self.W.data()
        for i, param in enumerate(params):
            length = (param ** 2).sum().asscalar()
            for j in range(len(param)):
                param[j] /= sqrt(length)

    def forward(self, X):
        output = nd.dot(X, self.W.data().T)
        output = nd.max(output)
        print(output)
        return -output


data_frame = pd.read_csv('西瓜3.0alpha.txt', sep=' ', encoding='gb2312')
data_frame = (data_frame - data_frame.mean()) / data_frame.std()
column_name = data_frame.columns
feature = data_frame[['密度', '含糖率']].as_matrix().astype('float32')

for x in feature:
    length = (x ** 2).sum()
    for j in range(len(x)):
        x[j] /= sqrt(length)

dataset = gdata.ArrayDataset(feature)
dataiter = gdata.DataLoader(dataset, batch_size=1)

som_net = Som(2, 5)
som_net.initialize()
trainer = gluon.Trainer(som_net.collect_params(), 'sgd', {'learning_rate': 10})

num_epochs = 100

for i in range(num_epochs):
    for x in dataiter:
        som_net.unify()
        with mxnet.autograd.record():
            loss = som_net(x)
        loss.backward()
        trainer.step(batch_size=1)
    print(loss.mean().asscalar())

som_net.unify()
for x in feature:
    plt.plot(*x, 'r.')
for center in som_net.W.data():
    plt.plot(center[0].asscalar(), center[1].asscalar(), 'b.')
plt.plot()
