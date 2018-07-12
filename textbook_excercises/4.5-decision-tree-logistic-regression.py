import pandas as pd
import numpy as np
from math import log

# 定义数据
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')

columns = data_frame.columns
discrete_columns = list(columns[1:-3])                    # 离散属性
label_column = '好瓜'                # label
label_map = {'是': 1, '否': 0}
data_frame[label_column] = data_frame[label_column].map(label_map)
data_frame = pd.get_dummies(data_frame, columns=discrete_columns)
feature_columns = [
    column for column in data_frame.columns if column not in ['编号', '好瓜']]


class Node:
    # 决策树的结点
    def __init__(self):
        self.positive_child = None
        self.negative_child = None
        self.label = None


import mxnet as mx
from mxnet import gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import ndarray as nd

sigmoid = mx.gluon.nn.Activation('sigmoid')


def build_decision_tree(data_frame, feature_columns, label_column):
    # print(data_frame)
    # 构造决策树
    node = Node()

    # 单独取出标记列
    label_series = list(data_frame[label_column].iteritems())
    # 如果所有样本标记相同,那么直接将该结点定义为叶节点
    if False not in [label_series[0][1] == label_series[i][1] for i in range(len(label_series))]:
        node.label = label_series[0][1]
        return node

    # 对每一个label取值进行计数
    most_y = 0
    most_cnt = 0
    cnt = {}
    for i, y in label_series:
        if y not in cnt.keys():
            cnt[y] = 1
        else:
            cnt[y] += 1
        if cnt[y] > most_cnt:
            most_y = y
            most_cnt = cnt[y]
    # 将当前结点标记为出现较多的label
    node.label = most_y

    # 根据当前结点的数据训练对率回归线性模型
    # 定义数据集
    dataset = gluon.data.ArrayDataset(
        data_frame[feature_columns].as_matrix().astype('float32'), data_frame[[label_column]].as_matrix().astype('float32'))
    dataiter = gdata.DataLoader(
        dataset, batch_size=len(data_frame), shuffle=False)

    # 定义网络
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.collect_params().initialize(init=mx.init.Normal())

    # 定义训练器
    trainer = gluon.Trainer(net.collect_params(),
                            'rmsprop', {'learning_rate': .5,'gamma1': .9})
    losser = gluon.loss.LogisticLoss(label_format='binary')

    # 训练模型
    num_epochs = 10
    verbose_epoch = 0
    for epoch in range(num_epochs):
        for feature, label in dataiter:
            with mx.autograd.record():
                label_hat = net(feature)
                # print(sigmoid.forward(label_hat))
                loss = losser(label_hat, label)
            loss.backward()
            trainer.step(len(data_frame))
        if epoch > verbose_epoch:
            print('Epoch', epoch, ', Loss =', loss.mean().asscalar())

    # 根据对率回归结果划分数据集
    positive_rows = []
    negative_rows = []
    for index, data in data_frame.iterrows():
        feature = nd.array(
            data[feature_columns].as_matrix()).reshape(shape=(1, -1))
        label = net(feature)[0][0]
        label = sigmoid.forward(label)
        print(label.asscalar(), data[label_column])
        if label > .5:
            positive_rows.append(index)
        else:
            negative_rows.append(index)

    # 递归建立子结点
    if len(positive_rows) == 0:
        node.label = 0
    elif len(negative_rows) == 0:
        node.label = 1
    else:
        node.positive_child = build_decision_tree(
            data_frame.loc[positive_rows], feature_columns, label_column)
        node.negative_child = build_decision_tree(
            data_frame.loc[negative_rows], feature_columns, label_column)
    return node


node = build_decision_tree(
    data_frame, feature_columns=feature_columns, label_column=label_column)
