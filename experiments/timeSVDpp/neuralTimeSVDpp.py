from mxnet import gluon
import mxnet
from load_movie_data import loadMovieData
from mxnet import ndarray as nd
import math
import random

# data_loader = loadMovieData()
# userItems, nUsers, nItems, nDays, minTimestamp = \
#     data_loader.main('ml-100k/u1.base')
# test_userItems, test_nUsers, test_nItems, test_nDays, test_minTimestapm = \
#     data_loader.main('ml-100k/u1.test')

# # 计算总平均分数
# average_rating = 0
# rating_cnt = 0
# for user in userItems.keys():
#     items = userItems[user]
#     for item in items:
#         average_rating += item[1]
#         rating_cnt += 1
# average_rating /= rating_cnt

# # 计算测试集容量
# test_rating_cnt = 0
# for user in test_userItems.keys():
#     items = test_userItems[user]
#     for item in items:
#         test_rating_cnt += 1

# # 为每一个用户计算他所有评分日的平均数
# user_meanday = {}
# for user in userItems.keys():
#     items = userItems[user]
#     meanday = 0
#     for item in items:
#         meanday += item[2] / len(items)
#     user_meanday[user] = meanday

# # 将数据整理到一个序列中
# data = []
# for user in userItems.keys():
#     for item in userItems[user]:
#         data.append((user,) + item)


class TimeSVDpp(gluon.nn.Block):

    def __init__(self, items_of_user, user_meanday,
                 item_cnt, user_cnt, day_cnt, average_rating,
                 factor_cnt=10, bin_cnt=30, beta=.4, ** kwargs):
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
        self.bin_size = math.ceil(day_cnt / bin_cnt)
        self.beta = beta

        with self.name_scope():
            # 设定学习参数q与y,即物品属性与使用该物品的用户的属性
            self.q = []
            self.y = []
            for i in range(item_cnt):
                self.q.append(self.params.get('q' + str(i), shape=(factor_cnt)))
                self.y.append(self.params.get('y' + str(i), shape=(factor_cnt)))

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
            self.p_long = []
            self.p_short = []
            self.alpha_preference = []
            for i in range(user_cnt):
                self.p_long.append(self.params.get(
                    'p_long' + str(i), shape=(factor_cnt)))
                self.p_short.append(self.params.get(
                    'p_short', shape=(day_cnt, factor_cnt)))
                self.alpha_preference.append(self.params.get(
                    'alpha_preference', shape=(factor_cnt)))

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
            self.mlp.add(gluon.nn.Dense(factor_cnt // 2, 'relu'))
            self.mlp.add(gluon.nn.Dropout(.5))
            # self.mlp.add(gluon.nn.Dense(factor_cnt // 4, 'relu'))
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
        q_i = self.q[i].data()
        y_i = self.y[i].data()
        # 取出物品偏移参数
        b_item_long = self.b_item_long.data()
        b_item_short = self.b_item_short.data()
        # 取出用户偏移参数
        b_user_long = self.b_user_long.data()
        alpha_bias = self.alpha_bias.data()
        b_user_short = self.b_user_short.data()
        # 取出用户偏好参数
        p_long_u = self.p_long[u].data()
        p_short_u = self.p_short[u].data()
        alpha_preference_u = self.alpha_preference[u].data()
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
        p = p_long_u + alpha_preference_u * \
            dev + p_short_u[t]                      # 用户偏好

        # 求用户评分过的商品所蕴含的用户偏好
        sum_y = nd.zeros(shape=(self.factor_cnt))
        for item in self.items_of_user[u]:
            item = item[0]
            sum_y = sum_y + y_i
        sum_y = sum_y / math.sqrt(len(self.items_of_user[u]))

        # 预测评分
        # relu = gluon.nn.Activation('relu')
        q = q_i.reshape((-1, 1))
        p = p.reshape((-1, 1)) + sum_y.reshape((-1, 1))
        qp = nd.concat(q, p, dim=0)
        # phi = nd.concat(self.mlp(qp.T).T, q * p, dim=0)
        phi = self.mlp(qp.T).T + nd.dot(q.T, p)
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

    # # 计算正则化项的值,也就是所有相关参数的平方和
    # def get_regularization(self, u, i, t):
    #     # 取出物品属性参数
    #     q_i = self.q[i].data()
    #     y_i = self.y[i].data()
    #     # 取出物品偏移参数
    #     b_item_long = self.b_item_long.data()
    #     b_item_short = self.b_item_short.data()
    #     # 取出用户偏移参数
    #     b_user_long = self.b_user_long.data()
    #     alpha_bias = self.alpha_bias.data()
    #     b_user_short = self.b_user_short.data()
    #     # 取出用户偏好参数
    #     p_long_u = self.p_long[u].data()
    #     p_short_u = self.p_short[u].data()
    #     alpha_preference_u = self.alpha_preference[u].data()
    #     # # 取出MLP参数
    #     # W1 = self.W1.data()
    #     # W2 = self.W2.data()
    #     # bW1 = self.bW1.data()
    #     # bW2 = self.bW2.data()
    #     # V1 = self.V1.data()
    #     # V2 = self.V2.data()
    #     # bV1 = self.bV1.data()
    #     # bV2 = self.bV2.data()

    #     regularization = (q_i ** 2).sum()
    #     regularization = regularization + (y_i ** 2).sum()
    #     regularization = regularization + b_item_long[i] ** 2
    #     regularization = regularization + \
    #         b_item_short[i][self._get_bin_index(t)] ** 2
    #     regularization = regularization + b_user_long[u] ** 2
    #     regularization = regularization + alpha_bias[u] ** 2
    #     regularization = regularization + b_user_short[u][t] ** 2
    #     regularization = regularization + (p_long_u ** 2).sum()
    #     regularization = regularization + \
    #         (alpha_preference_u ** 2).sum()
    #     regularization = regularization + (p_short_u ** 2).sum()
    #     for key in self.mlp.collect_params():
    #         regularization = regularization + \
    #             (self.mlp.collect_params()[key].data() ** 2).sum()
    #     # regularization = regularization + (W1 ** 2).sum()
    #     # regularization = regularization + (W2 ** 2).sum()
    #     # regularization = regularization + (bW1 ** 2).sum()
    #     # regularization = regularization + (bW2 ** 2).sum()
    #     # regularization = regularization + (V1 ** 2).sum()
    #     # regularization = regularization + (V2 ** 2).sum()
    #     # regularization = regularization + (bV1 ** 2).sum()
    #     # regularization = regularization + (bV2 ** 2).sum()
    #     return regularization


# 定义模型
timeSVDpp = TimeSVDpp(userItems, user_meanday,
                      nItems, nUsers, nDays, average_rating)
timeSVDpp.initialize()


class Trainer():
    # 记录所有数据
    def __init__(self, items_of_user, test_rating_cnt,
                 test_rating_cnt, rating_cnt):
        self.items_of_user = items_of_user
        self.test_rating_cnt = test_rating_cnt
        
        return 0

    # 测试模型
    def test(self):
        test_total_loss = 0
        tested_cnt = 0
        # 遍历测试集检验结果
        for user in test_userItems.keys():
            for item in test_userItems[user]:
                r_hat = timeSVDpp(user, item[0], int(item[2]))
                loss = (r_hat - item[1]) ** 2
                test_total_loss += loss
                tested_cnt += 1
                # 输出当前进度和误差
                print('Testing, tested percent:',
                      tested_cnt / test_rating_cnt, '. \tLoss =',
                      math.sqrt((test_total_loss / tested_cnt)[0].asscalar()),
                      end='\r')
        # 输出总误差
        print('\nTest finished, Loss =',
              math.sqrt((test_total_loss / test_rating_cnt)[0].asscalar()))

    # 训练模型
    def train(self, epoch_cnt=10, lambda_reg=.002, learning_method='sgd',
              learning_params=None, random_data=True):
        # 默认学习参数
        if learning_params is None:
            if learning_method == 'sgd':
                learning_params = {'learning_rate': .005}
            elif learning_method == 'adam':
                learning_params = {'beta1': .9, 'beta2': .999}
        learning_params['wd'] = lambda_reg

        # 定义训练器
        trainer = gluon.Trainer(timeSVDpp.collect_params(),
                                learning_method, learning_params)

        # 训练过程
        for epoch in range(epoch_cnt):
            total_loss = 0
            trained_cnt = 0
            # 使用随机数据训练
            if random_data is True:
                # 随机重排序列防止过拟合
                random.shuffle(data)
                # 训练模型
                for item in data:
                    with mxnet.autograd.record():
                        r_hat = timeSVDpp(item[0], item[1], int(item[3]))
                        loss = (r_hat - item[2]) ** 2
                        loss = loss + lambda_reg * \
                            timeSVDpp.get_regularization(item[0], item[1], int(item[3]))
                    loss.backward()
                    trainer.step(1, ignore_stale_grad=True)
                    total_loss += loss
                    trained_cnt += 1
                    print('Epoch', epoch, 'training, ', trained_cnt / rating_cnt,
                          'trained. \tLoss =',
                          (total_loss / trained_cnt)[0].asscalar(), end='\r')
            # 使用顺序数据训练
            else:
                for user in userItems:
                    items = userItems[user]
                    for item in items:
                        with mxnet.autograd.record():
                            r_hat = timeSVDpp(user, item[0], int(item[2]))
                            loss = (r_hat - item[1]) ** 2
                            loss = loss + lambda_reg * \
                                timeSVDpp.get_regularization(user, item[0], int(item[2]))
                        loss.backward()
                        trainer.step(1, ignore_stale_grad=True)
                        total_loss += loss
                        trained_cnt += 1
                        print('Epoch', epoch, 'training, ', trained_cnt / rating_cnt,
                              'trained. \tLoss =',
                              (total_loss / trained_cnt)[0].asscalar(), end='\r')
            print('\nEpoch', epoch, 'finished, Loss =',
                  (total_loss / rating_cnt)[0].asscalar())

            test()
