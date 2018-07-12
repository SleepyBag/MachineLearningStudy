from sklearn import naive_bayes
import pandas as pd

# import the data
data_frame = pd.read_csv('watermelon3_0_Ch.csv', encoding='gb2312')
label_column = '好瓜'
continuous_columns = ['密度', '含糖率']
discrete_columns = [column for column in data_frame.columns if column
                    not in continuous_columns and column != label_column]
continuous_feature = data_frame[continuous_columns]
discrete_feature = data_frame[discrete_columns]
discrete_feature = pd.get_dummies(discrete_feature)
label = data_frame[label_column]

indices = []

# for discrete_column in discrete_columns:
#     column_series = discrete_feature[discrete_column]
#     index = {}
#     index_cnt = 0
#     for data in column_series.iteritems():
#         data = data[1]
#         if data not in index.keys():
#             index[data] = index_cnt
#             index_cnt += 1
#     discrete_feature[discrete_column] = column_series.map(index)
#     indices.append(index)

continuous_feature = continuous_feature.as_matrix()
# discrete_feature = discrete_feature.as_matrix()

discrete_clfs = []
for column in discrete_columns:
    discrete_clf = naive_bayes.MultinomialNB(alpha=1.)
    discrete_clfs.append(discrete_clf)
    one_hoted_columns = [c for c in discrete_feature.columns
                         if c.startswith(column)]
    cur_feature = discrete_feature[one_hoted_columns]
    discrete_clf.fit(cur_feature[:17], label[:17])

continuous_clf = naive_bayes.GaussianNB()
continuous_clf.fit(continuous_feature[:17], label[:17])

# p_label_prior = continuous_clf.class_prior_
p_feature_priors = []
for i, column in enumerate(discrete_columns):
    p_feature_priors.append((discrete_clfs[i].feature_count_ / 17).tolist())
p_continuous_condition = continuous_clf.predict_proba(continuous_feature[-1:])
p_discrete_condition = []

for i, column in enumerate(discrete_columns):
    one_hoted_columns = [c for c in discrete_feature.columns
                         if c.startswith(column)]
    cur_feature = discrete_feature[one_hoted_columns]
    p_discrete_condition.append(
        discrete_clfs[i].predict_proba(cur_feature[-1:]).tolist())

for cur_label in range(2):
# p_post = p_label_prior * p_continuous_condition * p_discrete_condition
p_post = p_post.tolist()

prediction = p_post[0].index(max(p_post[0]))
prediction = discrete_clf.classes_[prediction]
