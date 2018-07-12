from sklearn import naive_bayes
import pandas as pd
import math

# import the data
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')
label_column = '好瓜'
continuous_columns = ['密度', '含糖率']
discrete_columns = [column for column in data_frame.columns if column
                    not in continuous_columns and column != label_column]
all_continuous_feature = data_frame[continuous_columns]
all_discrete_feature = data_frame[discrete_columns]
all_label = data_frame[label_column]

discrete_feature = [all_discrete_feature[:8], all_discrete_feature[8:-1]]
continuous_feature = [all_continuous_feature[:8], all_continuous_feature[8:-1]]

p_discrete_condition = [{}, {}]
ni = {}

for column in all_discrete_feature:
    ni[column] = 0
    dc = set()
    for data in all_discrete_feature[column]:
        if data not in dc:
            dc.add(data)
            ni[column] += 1

for c, pxc in zip(discrete_feature, p_discrete_condition):
    for column in c:
        new_data = set()
        for data in c[column]:
            if data in pxc:
                pxc[data] += 1
            else:
                pxc[data] = 1
                new_data.add(data)
        for key in new_data:
            pxc[key] += 1
            pxc[key] /= len(c) + ni[column]

p_prior = [len(c) / 17 for c in discrete_feature]
mu = [c.mean().tolist() for c in continuous_feature]
sigma = [c.std().tolist() for c in continuous_feature]


def p_continuous_condition(xi, c, i):
    cur_mu = mu[c][i]
    cur_sigma = sigma[c][i]
    return 1 / math.sqrt(2 * math.pi) / cur_sigma * \
        math.exp(-(xi - cur_mu) ** 2 / 2 / cur_sigma ** 2)


test_discrete_feature = all_discrete_feature[-1:]
test_continuous_feature = all_continuous_feature[-1:]

for c in range(2):
    p = p_prior[c]
    for column in test_discrete_feature:
        p *= p_discrete_condition[c][test_discrete_feature[column].iloc[0]]
    for i, column in enumerate(test_continuous_feature):
        p *= p_continuous_condition(
            test_continuous_feature[column].iloc[0], c, i)
    print(p)
