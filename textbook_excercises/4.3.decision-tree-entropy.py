import pandas as pd
import numpy as np
from math import log

# 定义数据
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')

columns = data_frame.columns
discrete_columns = list(columns[1:-3])                    # 离散属性
continuous_columns = list(columns[-3:-1])                 # 连续属性
label_column = columns[-1]                        # label


class Node:
    # 决策树的结点
    def __init__(self):
        self.child = {}
        self.label = None
        self.split_column = None
        self.split_point = None


def split_data_by_column(data_frame, split_column_name, split_discrete=True):
    # 根据某一列的属性,将数据分为属性不同的几组

    splited_data = []

    # 根据离散属性分割
    if split_discrete:
        # 单独取出要处理的列
        column = data_frame[split_column_name]

        # 将不同的属性值记录在dic的key中,每个key对应该值相应的行数
        dic = {}
        for data in column.iteritems():
            if data[1] not in dic.keys():
                dic[data[1]] = [data[0]]
            else:
                dic[data[1]].append(data[0])

        # 分割数据
        for key in dic.keys():
            rows = dic[key]
            splited_data.append(data_frame.loc[rows])

    # 根据连续属性分割
    else:
        data_frame = data_frame.sort_values(by=split_column_name)
        best_split_point = 0
        minimal_entropy = -1
        # 尝试所有分割点,对比熵的大小
        for split_point in range(1, len(data_frame)-1):
            splited_data = [data_frame.iloc[:split_point],
                            data_frame.iloc[split_point:]]
            if minimal_entropy == -1 or get_list_entropy(splited_data) < minimal_entropy:
                best_split_point = split_point
                minimal_entropy = get_list_entropy(splited_data)

        # 选择熵最小的分割点作为最终的分割点
        splited_data = [data_frame.iloc[:best_split_point],
                        data_frame.iloc[best_split_point:]]
        best_split_point = data_frame.iloc[best_split_point][split_column_name]

    if split_discrete:
        return splited_data
    else:
        return (splited_data, best_split_point)


def get_list_entropy(splited_data):
    # 求所有分支的熵之和

    len_sum = 0
    entropy = 0
    for data in splited_data:
        len_sum += len(data)
        entropy += get_entropy(data) * len(data)
    return entropy / len_sum


def get_entropy(splited_data):
    # 求该分支的熵之和
    # 对每一个label取值进行计数
    cnt = {}
    for y in splited_data[label_column]:
        if y not in cnt.keys():
            cnt[y] = 1
        else:
            cnt[y] += 1
    n = len(splited_data)

    # 统计熵值
    entropy = 0
    for key in cnt.keys():
        pk = cnt[key] / n
        entropy += -pk * log(pk) / log(2)
    return entropy


def build_decision_tree(data_frame, discrete_columns, continuous_columns, label_column):
    print(data_frame)
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
    node.label = most_y

    entropy = {column: 0 for column in list(
        discrete_columns)+list(continuous_columns)}
    # split_point = {column: 0 for column in list(
    # discrete_columns)+list(continuous_columns)}

    # 尝试以每一个离散属性为当前分支属性
    for column in discrete_columns:
        splited_data = split_data_by_column(data_frame, column)
        # 求每一个分支的熵
        entropy[column] = get_list_entropy(splited_data)

    # 尝试以每一个连续属性为当前分支属性
    for column in continuous_columns:
        splited_data, cur_split_point = split_data_by_column(
            data_frame, column, split_discrete=False)
        entropy[column] = get_list_entropy(splited_data)
        # split_point[column] = cur_split_point

    # 选择分支后熵最小的属性来分支
    best_split_column = ''
    minimal_entropy = -1
    for key in entropy.keys():
        if minimal_entropy == -1 or entropy[key] < minimal_entropy:
            minimal_entropy = entropy[key]
            best_split_column = key

    # 先判断分支属性是离散的还是连续的
    split_discrete = best_split_column in discrete_columns
    if split_discrete:
        splited_data = split_data_by_column(
            data_frame, best_split_column, split_discrete)
        discrete_columns.remove(best_split_column)
    else:
        splited_data, node.split_point = split_data_by_column(
            data_frame, best_split_column, split_discrete)
        continuous_columns.remove(best_split_column)

    # 根据最优分支构造子结点
    for data in splited_data:
        node.split_column = best_split_column
        node.child[data[best_split_column].iloc[0]] = build_decision_tree(
            data, [i for i in discrete_columns], [i for i in continuous_columns], label_column)

    return node


node = build_decision_tree(data_frame, discrete_columns=discrete_columns,
                           continuous_columns=continuous_columns, label_column=label_column)
