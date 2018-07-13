import pandas as pd
import numpy as np

# import the data
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')
label_column = '好瓜'
continuous_columns = ['密度', '含糖率']
discrete_columns = [column for column in data_frame.columns if column
                    not in continuous_columns and column != label_column]
all_discrete_feature = data_frame[discrete_columns]
all_label = data_frame[label_column]

train_discrete_feature = all_discrete_feature[:-1].as_matrix()
train_label = all_label[:-1].as_matrix()
test_discrete_feature = all_discrete_feature[-1:].as_matrix()
test_label = all_label[-1:].as_matrix()


class Spode:

    def __init__(self, feature, label, super_parent):
        # count the number of samples
        self.length = len(feature)
        self.p_joint = {}
        self.p_prior_c_pa = {}
        self.p_prior_c = {}
        self.n = {}
        self.c_range = self.p_prior_c.keys()

        # 统计各种概率
        for x, ci, pai in zip(feature, label, super_parent):
            # 标记先验统计
            if ci not in self.p_prior_c.keys():
                self.p_prior_c[ci] = 1
            else:
                self.p_prior_c[ci] += 1

            # 标记与超父的先验联合概率统计
            if (ci, pai) not in self.p_prior_c_pa.keys():
                self.p_prior_c_pa[(ci, pai)] = 1
            else:
                self.p_prior_c_pa[(ci, pai)] += 1

            for xi in x:
                # 联合概率统计
                if (xi, ci, pai) not in self.p_joint.keys():
                    self.p_joint[(xi, ci, pai)] = 1
                else:
                    self.p_joint[(xi, ci, pai)] += 1
        # 统计各种属性的取值可能数
        for Xi in feature.T:
            Xi = np.unique(Xi)
            for x in Xi:
                self.n[x] = len(Xi)
        Xi = np.unique(super_parent)
        for x in Xi:
            self.n[x] = len(Xi)

    def predict(self, feature, super_parent):
        ans = {}
        # 对所有可能的标记逐个分析
        for c in self.c_range:
            # 先验概率
            if (c, super_parent) in self.p_prior_c_pa.keys():
                ans[c] = self.p_prior_c_pa[(c, super_parent)] + 1
            else:
                ans[c] = 1
            ans[c] /= self.length + len(self.c_range) * self.n[super_parent]
            # 条件概率
            for xi in feature:
                if (xi, c, super_parent) in self.p_joint.keys():
                    ans[c] *= self.p_joint[(xi, c, super_parent)] + 1
                if (c, super_parent) in self.p_prior_c_pa.keys():
                    ans[c] /= self.p_prior_c_pa[(c, super_parent)] + self.n[xi]
                else:
                    ans[c] /= self.n[xi]
        return ans


class Aode:

    def __init__(self, feature, label):
        self.spodes = []
        feature = feature.T
        for i in range(len(feature)):
            super_parent = feature[i]
            cur_feature = np.delete(feature, i, 0).T
            self.spodes.append(Spode(cur_feature, label, super_parent))

    def predict(self, feature):
        ans = []
        feature = feature.T
        for i in range(len(feature)):
            super_parent = feature[i]
            cur_feature = np.delete(feature, i, 0).T
            ans.append(self.spodes[i].predict(cur_feature[0], super_parent[0]))
        return ans
