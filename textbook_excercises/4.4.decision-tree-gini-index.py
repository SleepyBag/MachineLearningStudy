import pandas as pd
import numpy as np
from math import log

# 定义数据
data_frame = pd.read_csv('西瓜2.0.txt', encoding='gb2312')
train_rows = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
test_rows = [i for i in range(len(data_frame)) if i not in train_rows]
train_data_frame = data_frame.iloc[train_rows]
test_data_frame = data_frame.iloc[test_rows]

columns = train_data_frame.columns
discrete_columns = list(columns[1:-1])  # 离散属性
continuous_columns = []                # 连续属性
label_column = columns[-1]             # label


class Node:
    # 决策树的结点
    def __init__(self, data_frame, label_column):
        self.child = {}
        self.split_column = None
        self.split_point = None

        label_series = list(data_frame[label_column].iteritems())
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
        # 将当前结点的标记设置为结点中占比最大的标记
        self.label = most_y

    def __str__(self):
        ans = str(self.child) + '\n'
        ans += str(self.label) + '\n'
        ans += str(self.split_column) + '\n'
        ans += str(self.split_point) + '\n'
        return ans

    def get_label(self, data_piece):
        # 取得一条数据的标记
        # 如果这个结点没有分支,则直接返回结点的标记
        if self.split_column == None:
            return self.label
        else:
            split_value = data_piece[self.split_column]
            # 如果结点有分支但数据的取值没有在分支中出现,则直接返回结点的标记
            if split_value not in self.child.keys():
                return self.label
            # 如果结点有分支且数据的取值在分支中出现了,则递归返回该分支的标记
            else:
                return self.child[split_value].get_label(data_piece)


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
        min_gini = -1
        # 尝试所有分割点,对比熵的大小
        for split_point in range(1, len(data_frame)-1):
            splited_data = [data_frame.iloc[:split_point],
                            data_frame.iloc[split_point:]]
            if min_gini == -1 or get_list_gini(splited_data) < min_gini:
                best_split_point = split_point
                min_gini = get_list_gini(splited_data)

        # 选择熵最小的分割点作为最终的分割点
        splited_data = [data_frame.iloc[:best_split_point],
                        data_frame.iloc[best_split_point:]]
        best_split_point = data_frame.iloc[best_split_point][split_column_name]

    if split_discrete:
        return splited_data
    else:
        return (splited_data, best_split_point)


def get_list_gini(splited_data):
    # 求所有分支的基尼指数之和

    len_sum = 0
    gini = 0
    for data in splited_data:
        len_sum += len(data)
        gini += get_gini(data) * len(data)
    return gini / len_sum


def get_gini(splited_data):
    # 求该分支的基尼指数

    # 对每一个label取值进行计数
    cnt = {}
    for y in splited_data[label_column]:
        if y not in cnt.keys():
            cnt[y] = 1
        else:
            cnt[y] += 1
    n = len(splited_data)

    # 统计基尼指数
    gini = 1
    for key in cnt.keys():
        pk = cnt[key] / n
        gini += - pk ** 2
    return gini


def build_decision_node(node, data_frame, discrete_columns, continuous_columns, label_column):
    # print(data_frame[discrete_columns + continuous_columns + [label_column]])
    # 构造决策树

    # 单独取出标记列
    label_series = list(data_frame[label_column].iteritems())
    # 如果所有样本标记相同,那么直接将该结点定义为叶节点
    if False not in [label_series[0][1] == label_series[i][1] for i in range(len(label_series))]:
        # node.label = label_series[0][1]
        return node

    # 如果没有可以分支的属性,那么该结点为叶节点

    gini = {column: 0 for column in list(
        discrete_columns)+list(continuous_columns)}
    # split_point = {column: 0 for column in list(
    # discrete_columns)+list(continuous_columns)}

    # 尝试以每一个离散属性为当前分支属性
    for column in discrete_columns:
        splited_data = split_data_by_column(data_frame, column)
        # 求每一个分支的熵
        gini[column] = get_list_gini(splited_data)

    # 尝试以每一个连续属性为当前分支属性
    for column in continuous_columns:
        splited_data, cur_split_point = split_data_by_column(
            data_frame, column, split_discrete=False)
        gini[column] = get_list_gini(splited_data)
        # split_point[column] = cur_split_point

    # 选择分支后熵最小的属性来分支
    best_split_column = None
    min_gini = -1
    # print(gini)
    for key in gini.keys():
        if min_gini == -1 or gini[key] <= min_gini:
            min_gini = gini[key]
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
    # print(splited_data)
    for data in splited_data:
        node.split_column = best_split_column
        node.child[data[best_split_column].iloc[0]] = build_decision_node(
            Node(data, label_column), data, [i for i in discrete_columns], [i for i in continuous_columns], label_column)

    return node


def build_decision_tree(data_frame, discrete_columns, continuous_columns, label_column):
    root = Node(data_frame, label_column)
    build_decision_node(root, data_frame, discrete_columns,
                        continuous_columns, label_column)
    return root


# 构造决策树
root = build_decision_tree(train_data_frame, discrete_columns=discrete_columns,
                           continuous_columns=continuous_columns, label_column=label_column)


def dfs(node):
    # 深度优先遍历决策树
    print(node)
    for child in node.child.keys():
        dfs(node.child[child])


# 遍历并输出
dfs(root)


def test(root, test_data, label_column):
    # 用test_data对root为根的决策树进行测试
    n = len(test_data)
    correct = 0
    # 逐个获取决策树预测的标记
    for data_piece in test_data.iterrows():
        data_piece = data_piece[1]
        label_hat = root.get_label(data_piece)
        # 统计预测正确的标记的数量
        if label_hat == data_piece[label_column]:
            correct += 1
        # print(data_piece, label_hat)
        # print('\n\n')
    # 返回预测正确的标记的比例
    return correct / n
