from mxnet import gluon
import mxnet
from load_movie_data import loadMovieData
from mxnet import ndarray as nd
import math
import random


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
            self.p_short = self.params.get('p_short', shape=(day_cnt, factor_cnt))
            self.alpha_preference = self.params.get(
                'alpha_preference', shape=(factor_cnt))
            self.p_mlp = self.params.get('p_mlp', shape=(factor_cnt))

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
        return p, b_user, self.p_mlp.data()


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
            self.q_mlp = self.params.get('q_mlp', shape=(factor_cnt))

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
        return q_i, b_item, self.q_mlp.data()

    # def get_regularization(self, t):
    #     return (self.q.data() ** 2).sum() + self.b_item_long.data() ** 2 + \
    #         self.b_item_short.data()[self._get_bin_index(t)] ** 2

# 代表评过某个商品的用户的属性,每个商品一个属性


class Y(gluon.nn.Block):
    def __init__(self, factor_cnt=10, **kwargs):
        super(Y, self).__init__(**kwargs)

        self.factor_cnt = factor_cnt
        with self.name_scope():
            self.y = self.params.get('y', shape=(factor_cnt))

    def forward(self):
        return self.y.data()

    # def get_regularization(self):
    #     return (self.y.data() ** 2).sum()


class TimeSVDpp(gluon.nn.Block):
    def __init__(self, average_rating, factor_cnt=10, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

        # 复制数据到对象中
        self.mu = average_rating

        # 增加MLP层
        with self.name_scope():
            self.mlp1 = gluon.nn.Sequential()
            self.mlp1.add(gluon.nn.Dense(factor_cnt, activation='relu'))
            self.mlp1.add(gluon.nn.Dense(factor_cnt // 2, activation='relu'))
            self.mlp1.add(gluon.nn.Dropout(.5))
            self.mlp1.add(gluon.nn.Dense(1))

    def forward(self, b_u, p_u, b_i, q_i, sum_y, p_mlp, q_mlp):

        # 预测评分
        q_i = q_i.reshape((1, -1))
        p_u = p_u.reshape((1, -1)) + sum_y.reshape((1, -1))
        p_mlp = p_mlp.reshape((1, -1))
        q_mlp = q_mlp.reshape((1, -1))
        phi = nd.dot(q_i, p_u.T) + self.mlp1(nd.concat(p_mlp, q_mlp, dim=1))
        r_hat = phi + b_i.reshape((1, 1)) + b_u.reshape((1, 1)) + self.mu
        return r_hat


# 测试模型
def test(items_of_user, timeSVDpp, factor_cnt=10, verbose=False):
    test_total_loss = 0
    tested_cnt = 0

    # 遍历测试集检验结果
    for user in test_userItems.keys():
        # 获取隐式反馈影响因素
        with mxnet.autograd.record():
            sum_y = nd.zeros((factor_cnt))
            for y in items_of_user[user]:
                sum_y = sum_y + net_ys[y[0]]()

        # 对该用户评过的所有商品进行迭代
        for item in test_userItems[user]:
            u = user
            i = int(item[0])
            r = int(item[1])
            t = int(item[2])
            p_u, b_u, p_mlp = net_users[u](t)
            q_i, b_i, q_mlp = net_items[i](t)
            r_hat = timeSVDpp(b_u, p_u, b_i, q_i, sum_y, p_mlp, q_mlp)
            loss = (r_hat - r) ** 2
            test_total_loss += loss
            tested_cnt += 1
            if verbose is False:
                # 输出当前进度和误差
                print('Testing, tested percent:',
                      tested_cnt / test_rating_cnt, '. \tLoss =',
                      math.sqrt((test_total_loss / tested_cnt)[0].asscalar()),
                      end='\r')

    # 输出总误差
    loss = math.sqrt((test_total_loss / test_rating_cnt)[0].asscalar())
    print('\nTest finished, Loss =', loss)
    return loss


# 训练模型
def train(items_of_user, net_users, net_items, net_ys, timeSVDpp, factor_cnt=10,
          epoch_cnt=10, batch_size=1, lambda_reg=.01, learning_method='sgd',
          learning_params=None, auto_stop=False, verbose=False):
    # 设置默认学习参数
    if learning_params is None:
        if learning_method == 'sgd':
            learning_params = {'learning_rate': .001}
        elif learning_method == 'adam':
            learning_params = {'beta1': .9, 'beta2': .999}
    learning_params['wd'] = lambda_reg

    # 定义训练器
    trainer_user = {}
    trainer_item = {}
    trainer_y = {}
    for user in net_users:
        trainer_user[user] = gluon.Trainer(net_users[user].collect_params(),
                                           learning_method, learning_params)
    for item in net_items:
        trainer_item[item] = gluon.Trainer(net_items[item].collect_params(),
                                           learning_method, learning_params)
    for y in net_ys:
        trainer_y[y] = (gluon.Trainer(net_ys[y].collect_params(),
                                      learning_method, learning_params))
    trainer_timeSVDpp = gluon.Trainer(timeSVDpp.collect_params(),
                                      learning_method, learning_params)

    test_loss = []
    # 训练过程
    for epoch in range(epoch_cnt):
        total_loss = 0
        trained_cnt = 0
        # 依次选择随机用户
        users = []
        for user in items_of_user:
            users.append(user)
        random.shuffle(users)
        for user in users:
            items = items_of_user[user]
            # 获取隐式反馈影响因素
            with mxnet.autograd.record():
                sum_y = nd.zeros((factor_cnt))
                for y in items_of_user[user]:
                    sum_y = sum_y + net_ys[y[0]]()
            sum_y_no_grad = sum_y.detach()
            loss_y = nd.zeros((1, 1))

            # 对该用户评过的所有商品进行迭代
            for item in items:
                u = user
                i = int(item[0])
                r = int(item[1])
                t = int(item[2])
                with mxnet.autograd.record():
                    # 计算除y以外的误差
                    p_u, b_u, p_mlp = net_users[u](t)
                    q_i, b_i, q_mlp = net_items[i](t)
                    r_hat = timeSVDpp(b_u, p_u, b_i, q_i,
                                      sum_y_no_grad, p_mlp, q_mlp)
                    loss = (r_hat - r) ** 2
                    # # 加上正则化项
                    # loss = loss + lambda_reg * (timeSVDpp.get_regularization() +
                    #                             net_users[u].get_regularization(t) +
                    #                             net_items[i].get_regularization(t))
                    # 计算y的误差
                    p_u_no_grad = p_u.detach()
                    b_u_no_grad = b_u.detach()
                    q_i_no_grad = q_i.detach()
                    b_i_no_grad = b_i.detach()
                    p_mlp_no_grad = p_mlp.detach()
                    q_mlp_no_grad = q_mlp.detach()
                    r_hat = timeSVDpp(b_u_no_grad, p_u_no_grad, b_i_no_grad,
                                      q_i_no_grad, sum_y,
                                      p_mlp_no_grad, q_mlp_no_grad)
                    loss_y = loss_y + (r_hat - r) ** 2
                # 更新除y以外的参数
                loss.backward()
                trainer_user[u].step(1)
                trainer_item[i].step(1)
                trainer_timeSVDpp.step(1)

                total_loss += loss

            # 添加正则化项并更新参数y
            with mxnet.autograd.record():
                loss_y = loss_y / len(items)
                # for y in items_of_user[u]:
                #     loss_y = loss_y + lambda_reg * net_ys[y[0]].get_regularization()
            loss_y.backward()
            for y in items_of_user[u]:
                trainer_y[y[0]].step(1)

            if verbose is False:
                # 输出训练信息
                trained_cnt += len(items)
                print('Epoch', epoch, 'training, ', trained_cnt / rating_cnt,
                      'trained. \tLoss =', (total_loss / trained_cnt)[0].asscalar(),
                      end='\r')

        print('\nEpoch', epoch, 'finished, Training loss =',
              (total_loss / rating_cnt)[0].asscalar())

        test_loss.append(test(items_of_user, timeSVDpp, factor_cnt))
        if auto_stop is True:
            if epoch != 0 and test_loss[epoch] > test_loss[epoch - 1]:
                break


lambda_reg = .01

# 定义模型
net_users = {}
net_items = {}
net_ys = {}
for i in range(1, nUsers + 1):
    net_users[i] = User(nDays, user_meanday[i])
    net_users[i].initialize()
for i in range(1, nItems + 1):
    net_items[i] = Item(nDays)
    net_items[i].initialize()
    net_ys[i] = Y()
    net_ys[i].initialize()
timeSVDpp = TimeSVDpp(average_rating)
timeSVDpp.initialize()

train(userItems, net_users, net_items, net_ys, timeSVDpp, factor_cnt=10,
      epoch_cnt=50, auto_stop=True, verbose=False, lambda_reg=lambda_reg)

lambda_reg = .01

# 定义模型
net_users = {}
net_items = {}
net_ys = {}
for i in range(1, nUsers + 1):
    net_users[i] = User(nDays, user_meanday[i])
    net_users[i].initialize()
for i in range(1, nItems + 1):
    net_items[i] = Item(nDays)
    net_items[i].initialize()
    net_ys[i] = Y()
    net_ys[i].initialize()
timeSVDpp = TimeSVDpp(average_rating)
timeSVDpp.initialize()

train(userItems, net_users, net_items, net_ys, timeSVDpp, factor_cnt=10,
      epoch_cnt=50, auto_stop=True, verbose=False, learning_method=' dam',
      lambda_reg=lambda_reg)
