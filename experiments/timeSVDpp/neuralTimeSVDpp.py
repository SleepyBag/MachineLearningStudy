from mxnet import gluon
import mxnet
from load_movie_data import loadMovieData
from mxnet import ndarray as nd
import math

data_loader = loadMovieData()
userItems, nUsers, nItems, nDays, minTimestamp = \
    data_loader.main('ml-100k/u1.base')
test_userItems, test_nUsers, test_nItems, test_nDays, test_minTimestapm = \
    data_loader.main('ml-100k/u1.test')

# 计算总平均分数
average_rating = 0
rating_cnt = 0
for user in userItems.keys():
    items = userItems[user]
    for item in items:
        average_rating += item[1]
        rating_cnt += 1
average_rating /= rating_cnt

# 计算测试集容量
test_rating_cnt = 0
for user in test_userItems.keys():
    items = test_userItems[user]
    for item in items:
        test_rating_cnt += 1

# 为每一个用户计算他所有评分日的平均数
user_meanday = {}
for user in userItems.keys():
    items = userItems[user]
    meanday = 0
    for item in items:
        meanday += item[2] / len(items)
    user_meanday[user] = meanday


class TimeSVDpp(gluon.nn.Block):

    def __init__(self, items_of_user, user_meanday,
                 item_cnt, user_cnt, day_cnt, average_rating,
                 factor_cnt=20, bin_size=30, beta=.4, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

        item_cnt += 1
        user_cnt += 1
        # 复制数据到对象中
        self.factor_cnt = factor_cnt
        self.items_of_user = items_of_user
        self.user_meanday = user_meanday
        self.item_cnt = item_cnt
        self.user_cnt = user_cnt
        self.mu = average_rating

        self.beta = beta
        self.bin_size = bin_size
        # 一个物品所经历的时间被分为一块块小的时间段,每个时间段被叫作一个bin,这是bin的总数
        bin_cnt = day_cnt // bin_size + 1

        with self.name_scope():
            # 设定学习参数q与y,即物品属性与使用该物品的用户的属性
            self.q = self.params.get('q', shape=(item_cnt, factor_cnt))
            self.y = self.params.get('y', shape=(item_cnt, factor_cnt))

            # 设定学习参数b_item_long与b_item_short,即物品的长期偏移与临时偏移
            self.b_item_long = self.params.get('b_item_long', shape=(item_cnt))
            self.b_item_short = self.params.get(
                'b_item_short', shape=(item_cnt, bin_cnt))

            # 设定学习参数b_user_long,alpha_bias与b_user_short,即用户的长期偏移,用户偏移的长期变化系数,和用户的单日偏移
            self.b_user_long = self.params.get('b_user_long', shape=(user_cnt))
            self.alpha_bias = self.params.get('alpha_bias', shape=(user_cnt))
            self.b_user_short = self.params.get(
                'b_user_short', shape=(user_cnt, day_cnt))

            # 设定学习参数p_long,alpha_preference与p_short,即用户的偏好,用户偏好的长期变化系数,用户爱好的单日变化
            self.p_long = self.params.get(
                'p_long', shape=(user_cnt, factor_cnt))
            self.alpha_preference = self.params.get(
                'alpha_preference', shape=(user_cnt, factor_cnt))
            self.p_short = self.params.get(
                'p_short', shape=(user_cnt, day_cnt, factor_cnt))

            # # 设定MLP的学习参数W1,W2,V1,V2
            # self.W1 = self.params.get('W1', shape=(factor_cnt, 2 * factor_cnt))
            # self.bW1 = self.params.get('bW1', shape=(factor_cnt, 1))
            # self.W2 = self.params.get('W2', shape=(1, factor_cnt))
            # self.bW2 = self.params.get('bW2', shape=(1, 1))
            # self.V1 = self.params.get('V1', shape=(2, 3))
            # self.bV1 = self.params.get('bV1', shape=(2, 1))
            # self.V2 = self.params.get('V2', shape=(1, 2))
            # self.bV2 = self.params.get('bV2', shape=(1, 1))

            self.mlp = gluon.nn.Sequential()
            self.mlp.add(gluon.nn.Dense(factor_cnt, 'relu'))
            self.mlp.add(gluon.nn.Dropout(.5))
            self.mlp.add(gluon.nn.Dense(1, 'relu'))

    # 求某一个日期对某用户的平均评分日期的偏移
    def _get_dev(self, u, t):
        mean_day = user_meanday[u]
        deviation = t - mean_day
        return 0 if deviation == 0 else (deviation) / math.fabs(deviation) * \
            math.fabs(deviation) ** self.beta

    # 求一个日期对应的bin的编号
    def _get_bin_index(self, t):
        return t // self.bin_size

    def forward(self, u, i, t):
        # 取出物品属性参数
        q = self.q.data()
        y = self.y.data()
        # 取出物品偏移参数
        b_item_long = self.b_item_long.data()
        b_item_short = self.b_item_short.data()
        # 取出用户偏移参数
        b_user_long = self.b_user_long.data()
        alpha_bias = self.alpha_bias.data()
        b_user_short = self.b_user_short.data()
        # 取出用户偏好参数
        p_long = self.p_long.data()
        alpha_preference = self.alpha_preference.data()
        p_short = self.p_short.data()
        # # 取出MLP参数
        # W1 = self.W1.data()
        # bW1 = self.bW1.data()
        # W2 = self.W2.data()
        # bW2 = self.bW2.data()
        # V1 = self.V1.data()
        # bV1 = self.bV1.data()
        # V2 = self.V2.data()
        # bV2 = self.bV2.data()

        dev = self._get_dev(u, t)

        b_item = b_item_long[i] + \
            b_item_short[i, self._get_bin_index(t)]  # 物品偏移
        b_user = b_user_long[u] + alpha_bias[u] * \
            dev + b_user_short[u, t]                 # 用户偏移
        p = p_long[u] + alpha_preference[u] * \
            dev + p_short[u, t]                      # 用户偏好

        # 求用户评分过的商品所蕴含的用户偏好
        sum_y = nd.zeros(shape=(self.factor_cnt))
        for item in self.items_of_user[u]:
            item = item[0]
            sum_y = sum_y + y[item]
        sum_y = sum_y / math.sqrt(len(self.items_of_user[u]))

        # 预测评分
        # relu = gluon.nn.Activation('relu')
        q = q[i].reshape((-1, 1))
        p = p.reshape((-1, 1)) + sum_y.reshape((-1, 1))
        qp = nd.concat(q, p, dim=0)
        phi = self.mlp(qp) + nd.dot(q.T, p)
        r_hat = phi + b_item.reshape((1, 1)) + b_user.reshape((1, 1)) + self.mu
        # mW = nd.dot(W1, qp) + bW1
        # mW = relu(mW)
        # phi = (nd.dot(W2, mW) + bW2) + nd.dot(q.T, p)
        # preference = nd.concat(phi, b_item.reshape((1, 1)),
        #                        b_user.reshape((1, 1)), dim=0)
        # mV = nd.dot(V1, preference) + bV1
        # mV = relu(mV)
        # r_hat = (nd.dot(V2, mV) + bV2) + preference.sum()

        return r_hat

    # 计算正则化项的值,也就是所有相关参数的平方和
    def get_regularization(self, u, i, t):
        # 取出物品属性参数
        q = self.q.data()
        y = self.y.data()
        # 取出物品偏移参数
        b_item_long = self.b_item_long.data()
        b_item_short = self.b_item_short.data()
        # 取出用户偏移参数
        b_user_long = self.b_user_long.data()
        alpha_bias = self.alpha_bias.data()
        b_user_short = self.b_user_short.data()
        # 取出用户偏好参数
        p_long = self.p_long.data()
        alpha_preference = self.alpha_preference.data()
        p_short = self.p_short.data()
        # # 取出MLP参数
        # W1 = self.W1.data()
        # W2 = self.W2.data()
        # bW1 = self.bW1.data()
        # bW2 = self.bW2.data()
        # V1 = self.V1.data()
        # V2 = self.V2.data()
        # bV1 = self.bV1.data()
        # bV2 = self.bV2.data()

        regularization = (q[i] ** 2).sum()
        regularization = regularization + (y[i] ** 2).sum()
        regularization = regularization + b_item_long[i] ** 2
        regularization = regularization + \
            b_item_short[i][self._get_bin_index(t)] ** 2
        regularization = regularization + b_user_long[u] ** 2
        regularization = regularization + alpha_bias[u] ** 2
        regularization = regularization + b_user_short[u][t] ** 2
        regularization = regularization + (p_long[u] ** 2).sum()
        regularization = regularization + \
            (alpha_preference[u] ** 2).sum()
        regularization = regularization + (p_short[u] ** 2).sum()
        for key in self.mlp.collect_params():
            regularization = regularization + \
                (self.mlp.collect_params()[key].data() ** 2).sum()
        # regularization = regularization + (W1 ** 2).sum()
        # regularization = regularization + (W2 ** 2).sum()
        # regularization = regularization + (bW1 ** 2).sum()
        # regularization = regularization + (bW2 ** 2).sum()
        # regularization = regularization + (V1 ** 2).sum()
        # regularization = regularization + (V2 ** 2).sum()
        # regularization = regularization + (bV1 ** 2).sum()
        # regularization = regularization + (bV2 ** 2).sum()
        return regularization


timeSVDpp = TimeSVDpp(userItems, user_meanday,
                      nItems, nUsers, nDays, average_rating)
timeSVDpp.initialize()
# trainer = gluon.Trainer(timeSVDpp.collect_params(), 'sgd',
#                         {'learning_rate': .0005})
trainer = gluon.Trainer(timeSVDpp.collect_params(), 'adam',
                        {'beta1': .9, 'beta2': .999})

lambda_reg = .002
epoch_cnt = 10
for epoch in range(epoch_cnt):
    total_loss = 0
    trained_cnt = 0
    for user in userItems.keys():
        for item in userItems[user]:
            with mxnet.autograd.record():
                r_hat = timeSVDpp(user, item[0], int(item[2]))
                loss = (r_hat - item[1]) ** 2
                loss = loss + lambda_reg * \
                    timeSVDpp.get_regularization(user, item[0], int(item[2]))
            loss.backward()
            trainer.step(1)
            total_loss += loss
            trained_cnt += 1
            print('Epoch', epoch, 'training, ', trained_cnt / rating_cnt,
                  'trained. \tLoss =',
                  (total_loss / trained_cnt)[0].asscalar(), end='\r')
    print('\nEpoch', epoch, 'finished, Loss =',
          (total_loss / rating_cnt)[0].asscalar())

    test_total_loss = 0
    tested_cnt = 0
    for user in test_userItems.keys():
        for item in test_userItems[user]:
            r_hat = timeSVDpp(user, item[0], int(item[2]))
            loss = (r_hat - item[1]) ** 2
            test_total_loss += loss
            tested_cnt += 1
            print('Epoch', epoch, 'testing, tested percent:',
                  tested_cnt / test_rating_cnt, '. \tLoss =',
                  (test_total_loss / tested_cnt)[0].asscalar(), end='\r')
    print('\nTest finished, Loss =',
          (test_total_loss / test_rating_cnt)[0].asscalar())
