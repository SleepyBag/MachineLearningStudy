# coding:utf-8

from pomegranate import BayesianNetwork
import pandas as pd
import numpy as np

data_frame = pd.read_csv(
    '西瓜2.0.txt', names=['1', '2', '3', '4', '5', '6', '7'], encoding='gb2312')

for column in data_frame:
    index = {}
    cur_index = 0
    for data in data_frame[column]:
        if data not in index.keys():
            index[data] = cur_index
            cur_index += 1
    data_frame[column] = data_frame[column].map(index)

data = data_frame.as_matrix()
print(data)
bayes_net = BayesianNetwork.from_samples(data, name='watermelon')
bayes_net.plot('a.png')
