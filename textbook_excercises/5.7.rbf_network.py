import mxnet
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import data as gdata


# 径向基层
class Rbf(gluon.Block):
    # 初始化函数
    def __init__(self, data, **kwargs):
        super(Rbf, self).__init__(**kwargs)
        # use name_scope to give child Blocks appropriate names.
        # It also allows sharing Parameters between Blocks recursively.
        self.beta = self.params.get('beta', shape=(len(data), 1))
        self.C = data

    # 前向传递函数
    def forward(self, X):
        def rho(x, betai, ci):
            return nd.exp(-betai * ((x - ci) ** 2).sum())

        y = []
        for x in X:
            ans = []
            for beta, c in zip(self.beta.data(), self.C):
                ans.append(nd.array(rho(x, beta, c)).reshape((1, 1)))
            y.append(nd.concat(*ans, dim=1))
        return nd.concat(*y, dim=0)


feature = nd.array([[0, 0], [0, 1], [1, 0], [1, 1]])
label = nd.array([0, 1, 1, 0])
rbf_net = Rbf(feature)

net = nn.Sequential()
with net.name_scope():
    net.add(rbf_net)
    net.add(nn.Dense(1))
net.initialize()

dataset = gdata.ArrayDataset(feature, label)
dataiter = gdata.DataLoader(dataset, batch_size=4)
losser = gluon.loss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1})

num_epochs = 100
for epoch in range(num_epochs):
    for featurei, label in dataiter:
        with mxnet.autograd.record():
            label_hat = net(featurei)
            loss = losser(label_hat, label)
        loss.backward()
        trainer.step(batch_size=4)
        print(loss.mean().asscalar())
