import pandas as pd

# 定义数据
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')

columns = data_frame.columns
discrete_columns = list(columns[1:-3])      # 离散属性
continuous_columns = list(columns[-3:-1])   # 连续属性
label_column = columns[-1]                  # label

from mxnet import ndarray as nd

code_dict = {'是': 1, '否': 0}
feature = data_frame[discrete_columns + continuous_columns]
feature = nd.array(pd.get_dummies(
    feature, columns=discrete_columns).as_matrix())
label = data_frame[label_column].replace(code_dict).as_matrix()

from mxnet import autograd
from mxnet.gluon import nn
from mxnet import gluon
from mxnet.gluon import data as gdata


def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(len(discrete_columns + continuous_columns),
                         activation='sigmoid'))
        net.add(nn.Dense(2))
    return net


losser = gluon.loss.SoftmaxCrossEntropyLoss()

bp_batch_size = 1
abp_batch_size = len(feature)
dataset = gdata.ArrayDataset(feature, label)
bp_data_iter = gdata.DataLoader(dataset, bp_batch_size, shuffle=True)
abp_data_iter = gdata.DataLoader(dataset, abp_batch_size, shuffle=False)

bp_net = get_net()
bp_net.collect_params().initialize(gluon.parameter.initializer.Normal())

abp_net = get_net()
abp_net.collect_params().initialize(gluon.parameter.initializer.Normal())

bp_trainer = gluon.Trainer(bp_net.collect_params(),
                           'sgd', {'learning_rate': .005})
abp_trainer = gluon.Trainer(abp_net.collect_params(),
                            'sgd', {'learning_rate': .005})

num_epoch = 100

from matplotlib import pyplot as plt


bp_loss_list = []
abp_loss_list = []
for epoch in range(num_epoch):

    print('Epoch', epoch)
    bp_loss = 0
    abp_loss = 0
    for feature, label in bp_data_iter:
        with autograd.record():
            label_hat = bp_net(feature)
            loss = losser(label_hat, label)
        loss.backward()
        bp_loss += loss.asscalar()
        bp_trainer.step(bp_batch_size)

    for feature, label in abp_data_iter:
        with autograd.record():
            label_hat = abp_net(feature)
            loss = losser(label_hat, label)
        loss.backward()
        abp_loss += loss.sum()
        abp_trainer.step(abp_batch_size)
    bp_loss_list.append(bp_loss)
    abp_loss_list.append(abp_loss.asscalar())

plt.plot(range(num_epoch), bp_loss_list, 'r.')
plt.plot(range(num_epoch), abp_loss_list, 'b.')
plt.show()
