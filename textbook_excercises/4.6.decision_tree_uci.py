from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import graphviz

# 将数据从csv文件中读取到dataframe
data = pd.read_csv('iris.data', sep=',',
                   names=['sepal_length', 'sepal_width',
                          'petal_length', 'petal_width', 'class'])
# 打乱数据
data = data.sample(frac=1)


def k_fold(data, k):

    data_length = len(data)
    fold_length = data_length // k
    accuracy = 0
    for i in range(k):

        # 取数据所需要的坐标
        test_index = list(range(fold_length * i, fold_length * (i + 1)))
        train_index = [i for i in range(data_length) if i not in test_index]
        feature_names = ['sepal_length', 'sepal_width',
                         'petal_length', 'petal_width']
        # 将数据整理为feature与label
        train_feature = data.loc[train_index, feature_names].as_matrix()
        train_label = data.loc[train_index, 'class'].map(
            {'Iris-setosa': 0,
             'Iris-versicolor': 1,
             'Iris-virginica': 2}).as_matrix()
        test_feature = data.loc[test_index, feature_names].as_matrix()
        test_label = data.loc[test_index, 'class'].map(
            {'Iris-setosa': 0,
             'Iris-versicolor': 1,
             'Iris-virginica': 2}).as_matrix()

        # 训练决策树
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=-9999999999)
        decision_tree.fit(train_feature, train_label)

        # 为训练集进行预测
        test_label_hat = decision_tree.predict_proba(test_feature).tolist()
        test_label_hat = [label.index(max(label)) for label in test_label_hat]

        # 统计正确预测数
        for label, label_hat in zip(test_label, test_label_hat):
            if label == label_hat:
                accuracy += 1

        # # 绘制图形
        # dot_data = tree.export_graphviz(decision_tree, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("iris")
    accuracy /= data_length
    print(accuracy)


k_fold(data, len(data))
