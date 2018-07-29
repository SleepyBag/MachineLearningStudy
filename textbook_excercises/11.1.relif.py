import pandas as pd
import numpy as np
from math import log
import math

# 定义数据
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')

columns = data_frame.columns
discrete_columns = list(columns[0:-3])                    # 离散属性
continuous_columns = list(columns[-3: -1])              # 连续属性
discrete_feature = data_frame[discrete_columns]
continuous_feature = data_frame[continuous_columns]
continuous_feature = (continuous_feature - continuous_feature.min()) / \
    (continuous_feature.max() - continuous_feature.min())
data_frame[continuous_columns] = continuous_feature
label_column = '好瓜'                # label
feature_columns = [
    column for column in data_frame.columns if column not in ['编号', '好瓜']]


def diff(a, b, column):
    if column in discrete_columns:
        return 1 if a[column] == b[column] else 0
    else:
        return math.fabs(a[column] - b[column])


def distance(a, b):
    dis = 0
    for column in feature_columns:
        dis += diff(a, b, column)
    return dis


near_hit = {}
near_miss = {}

for piece in data_frame.iterrows():
    number = piece[0]
    data = piece[1]
    near_hit[number] = None
    near_miss[number] = None
    for tmp_piece in data_frame.iterrows():
        tmp_number = tmp_piece[0]
        tmp_data = tmp_piece[1]
        if tmp_data[label_column] == data[label_column]:
            if near_hit[number] is None or \
                    distance(data, tmp_data) < distance(data, near_hit[number]):
                near_hit[number] = tmp_data
        else:
            if near_miss[number] is None or \
                    distance(data, tmp_data) < distance(data, near_miss[number]):
                near_miss[number] = tmp_data

delta = {}
for column in feature_columns:
    delta[column] = 0
    for piece in data_frame.iterrows():
        number = piece[0]
        data = piece[1]
        nh = near_hit[number]
        nm = near_miss[number]
        delta[column] += diff(data, nm, column) ** 2 - diff(data, nh, column) ** 2
