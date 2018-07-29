from mxnet import gluon
import mxnet
from mxnet import ndarray as nd
import math
import random


# 用户的属性
class User(gluon.nn.Block):
    def __init__(self, day_cnt, meanday, factor_cnt=10, beta=.4, ** kwargs):
        super(User, self).__init__(**kwargs)

        # 复制数据到对象中
        self.factor_cnt = factor_cnt
        self.meanday = meanday
        self.beta = beta

        with self.name_scope():
            # 设定学习参数b_user_long,alpha_bias与b_user_short,即用户的长期偏移,用户偏移的长期变化系数,和用户的单日偏移
            self.b_user_long = self.params.get('b_user_long', shape=(1))
            self.alpha_bias = self.params.get('alpha_bias', shape=(1))
            self.b_user_short = self.params.get('b_user_short', shape=(day_cnt))

            # 设定学习参数p_long,alpha_preference与p_short,即用户的偏好,用户偏好的长期变化系数,用户爱好的单日变化
            self.p_long = self.params.get('p_long', shape=(factor_cnt))
            self.p_short = self.params.get(
                'p_short', shape=(day_cnt, factor_cnt))
            self.alpha_preference = self.params.get(
                'alpha_preference', shape=(factor_cnt))

    def _get_dev(self, t):
        # 求某一个日期对某用户的平均评分日期的偏移
        deviation = t - self.meanday
        return 0 if deviation == 0 else (deviation) / math.fabs(deviation) * \
            math.fabs(deviation) ** self.beta

    def forward(self, t):
        # 取出用户偏移参数
        b_user_long = self.b_user_long.data()
        alpha_bias = self.alpha_bias.data()
        b_user_short = self.b_user_short.data()
        # 取出用户偏好参数
        p_long_u = self.p_long.data()
        p_short_u = self.p_short.data()
        alpha_preference_u = self.alpha_preference.data()

        dev = self._get_dev(t)

        b_user = b_user_long + alpha_bias * dev + b_user_short[t]  # 用户偏移
        p = p_long_u + alpha_preference_u * dev + p_short_u[t]     # 用户偏好
        return p, b_user

    def get_regularization(self, t):
        return self.b_user_long.data() ** 2 + self.alpha_bias.data() ** 2 \
            + self.b_user_short.data()[t] ** 2 + (self.p_long.data() ** 2).sum() \
            + (self.p_short.data()[t] ** 2).sum() \
            + self.alpha_preference.data().sum()


# 商品的属性
class Item(gluon.nn.Block):
    def __init__(self, day_cnt, factor_cnt=10, bin_cnt=30, ** kwargs):
        super(Item, self).__init__(**kwargs)

        # 复制数据到对象中
        self.day_cnt = day_cnt
        self.factor_cnt = factor_cnt
        self.bin_cnt = bin_cnt
        self.bin_size = math.ceil(day_cnt / bin_cnt)

        with self.name_scope():
            # 设定学习参数q,即物品属性与使用该物品的用户的属性
            self.q = self.params.get('q', shape=(factor_cnt))

            # 设定学习参数b_item_long与b_item_short,即物品的长期偏移与临时偏移
            self.b_item_long = self.params.get('b_item_long', shape=(1))
            self.b_item_short = self.params.get('b_item_short', shape=(bin_cnt))

    # 求一个日期对应的bin的编号
    def _get_bin_index(self, t):
        return t // self.bin_size

    def forward(self, t):
        # 取出物品属性参数
        q_i = self.q.data()
        # 取出物品偏移参数
        b_item_long = self.b_item_long.data()
        b_item_short = self.b_item_short.data()

        b_item = b_item_long + b_item_short[self._get_bin_index(t)]  # 物品偏移
        return q_i, b_item

    def get_regularization(self, t):
        return (self.q.data() ** 2).sum() + self.b_item_long.data() ** 2 + \
            self.b_item_short.data()[self._get_bin_index(t)] ** 2

# 代表评过某个商品的用户的属性,每个商品一个属性


class Y(gluon.nn.Block):
    def __init__(self, factor_cnt=10, **kwargs):
        super(Y, self).__init__(**kwargs)

        self.factor_cnt = factor_cnt
        with self.name_scope():
            self.y = self.params.get('y', shape=(factor_cnt))

    def forward(self):
        return self.y.data()

    def get_regularization(self):
        return (self.y.data() ** 2).sum()


class TimeSVDpp(gluon.nn.Block):
    def __init__(self, average_rating, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

        # 复制数据到对象中
        self.mu = average_rating

        # 朴素TimeSVD++没有需要训练的参数
        with self.name_scope():
            pass

    def forward(self, b_u, p_u, b_i, q_i, y_ru):

        # 求用户评分过的商品所蕴含的用户偏好
        sum_y = nd.zeros(shape=p_u.shape)
        for y in y_ru:
            sum_y = sum_y + y
        sum_y = sum_y / math.sqrt(len(y_ru))

        # 预测评分
        q_i = q_i.reshape((-1, 1))
        p_u = p_u.reshape((-1, 1)) + sum_y.reshape((-1, 1))
        phi = nd.dot(q_i.T, p_u)
        r_hat = phi + b_i.reshape((1, 1)) + b_u.reshape((1, 1)) + self.mu
        return r_hat

    # 计算正则化项的值,也就是所有相关参数的平方和
    def get_regularization(self):
        return 0


# 测试模型
def test():
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


# 定义模型
users = {}
items = {}
ys = {}
for i in range(1, nUsers + 1):
    users[i] = User(nDays, user_meanday[i])
    users[i].initialize()
for i in range(1, nItems + 1):
    items[i] = Item(nDays)
    items[i].initialize()
    ys[i] = Y()
    ys[i].initialize()
timeSVDpp = TimeSVDpp(average_rating)
timeSVDpp.initialize()


# 训练模型
def train(items_of_user, users, items, ys, timeSVDpp, epoch_cnt=10,
          batch_size=1, lambda_reg=.01, learning_method='sgd',
          learning_params=None, random_data=True):
    # 设置默认学习参数
    if learning_params is None:
        if learning_method == 'sgd':
            learning_params = {'learning_rate': .005}
        elif learning_method == 'adam':
            learning_params = {'beta1': .9, 'beta2': .999}
    learning_params['wd'] = lambda_reg

    # 定义训练器
    trainer_user = {}
    trainer_item = {}
    trainer_y = {}
    for user in users:
        trainer_user[user] = gluon.Trainer(users[user].collect_params(),
                                           learning_method, learning_params)
    for item in items:
        trainer_item[item] = gluon.Trainer(items[item].collect_params(),
                                           learning_method, learning_params)
    for y in ys:
        trainer_y[y] = (gluon.Trainer(ys[y].collect_params(),
                                      learning_method, learning_params))

    # 训练过程
    for epoch in range(epoch_cnt):
        total_loss = 0
        trained_cnt = 0
        # 使用随机数据训练
        if random_data is True:
            # 随机重排序列防止过拟合
            random.shuffle(data)
            # 遍历数据集训练
            for item in data:
                u = int(item[0])
                i = int(item[1])
                r = int(item[2])
                t = int(item[3])
                with mxnet.autograd.record():
                    y_ru = []
                    for y in items_of_user[u]:
                        y_ru.append(ys[y[0]]())
                    p_u, b_u = users[u](t)
                    q_i, b_i = items[i](t)
                    r_hat = timeSVDpp(b_u, p_u, b_i, q_i, y_ru)
                    # 正则化参数
                    regularization = timeSVDpp.get_regularization() + \
                        users[u].get_regularization(t) + \
                        items[i].get_regularization(t)
                    for y in items_of_user[u]:
                        regularization = regularization + \
                            ys[y[0]].get_regularization()
                    loss = (r_hat - r) ** 2 + lambda_reg * regularization
                # 单步训练
                loss.backward()
                trainer_user[u].step(1)
                trainer_item[i].step(1)
                for y in items_of_user[u]:
                    trainer_y[y[0]].step(1)
                total_loss += loss
                trained_cnt += 1
                print('Epoch', epoch, 'training, ', trained_cnt / rating_cnt,
                      'trained. \tLoss =',
                      (total_loss / trained_cnt)[0].asscalar(), end='\r')
        # # 使用顺序数据训练
        # else:
        #     for user in userItems:
        #         items = userItems[user]
        #         for item in items:
        #             with mxnet.autograd.record():
        #                 r_hat = timeSVDpp(user, item[0], int(item[2]))
        #                 loss = (r_hat - item[1]) ** 2
        #                 loss = loss + lambda_reg * \
        #                     timeSVDpp.get_regularization(user, item[0], int(item[2]))
        #             loss.backward()
        #             trainer.step(1, ignore_stale_grad=True)
        #             total_loss += loss
        #             trained_cnt += 1
        #             print('Epoch', epoch, 'training, ', trained_cnt / rating_cnt,
        #                   'trained. \tLoss =',
        #                   (total_loss / trained_cnt)[0].asscalar(), end='\r')
        # print('\nEpoch', epoch, 'finished, Loss =',
        #       (total_loss / rating_cnt)[0].asscalar())

        test()


train(userItems, users, items, ys, timeSVDpp, learning_method='adam')
