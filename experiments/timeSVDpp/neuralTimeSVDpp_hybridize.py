from mxnet import gluon
import mxnet
from mxnet import ndarray as nd
import math
import random


class TimeSVDpp(gluon.nn.HybridBlock):

    def __init__(self, items_of_user, user_meanday,
                 item_cnt, user_cnt, day_cnt, average_rating,
                 factor_cnt=20, bin_cnt=30, beta=.4, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

        # 复制数据到对象中
        self.factor_cnt = factor_cnt
        self.items_of_user = items_of_user
        self.user_meanday = user_meanday
        self.item_cnt = item_cnt
        self.user_cnt = user_cnt
        self.mu = average_rating

        self.beta = beta
        self.bin_cnt = bin_cnt

        with self.name_scope():
            # 定义MLP块
            self.mlp1 = gluon.nn.HybridSequential()
            self.mlp1.add(gluon.nn.Dense(factor_cnt))
            self.mlp1.add(gluon.nn.Dropout())
            self.mlp1.add(gluon.nn.Dense(1))
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

            # 设定MLP模块的学习参数p_mlp和q_mlp,分别代表用户和商品(不考虑时间变化)
            self.p_mlp = self.params.get('p_mlp', shape=(user_cnt, factor_cnt))
            self.q_mlp = self.params.get('q_mlp', shape=(user_cnt, factor_cnt))

    def hybrid_forward(self, F, u, i, t, y_i, dev, bint, q, y, b_item_long,
                       b_item_short, b_user_long, alpha_bias, b_user_short,
                       p_long, alpha_preference, p_short, p_mlp, q_mlp):

        b_item = F.dot(i, b_item_long).reshape((1, -1)) + \
            F.dot(bint, F.transpose(F.dot(i, b_item_short))).reshape((1, -1))  # 物品偏移
        b_user = F.dot(u, b_user_long).reshape((1, -1)) + \
            F.dot(u, alpha_bias).reshape((1, -1)) * dev.reshape((1, 1)) + \
            F.dot(t, F.transpose(F.dot(u, b_user_short))).reshape((1, -1))     # 用户偏移
        p = F.dot(u, p_long) + F.dot(u, alpha_preference) * \
            F.broadcast_to(dev.reshape((1, 1)), (1, self.factor_cnt)) + \
            F.dot(t, F.sum(F.dot(u, p_short), axis=0)).reshape((1, -1))       # 用户偏好
        q_i = F.dot(i, q)
        sum_y = F.dot(y_i, y)
        p_u_mlp = F.dot(u, p_mlp)
        q_i_mlp = F.dot(i, q_mlp)
        phi = F.dot(F.transpose(q_i.reshape((-1, 1))),
                    p.reshape((-1, 1)) + sum_y.reshape((-1, 1))) + \
            self.mlp1(F.concat((p_u_mlp, q_i_mlp), dim=1))

        # 预测评分
        r_hat = self.mu + b_item + b_user + phi

        return r_hat


# 求某一个日期对某用户的平均评分日期的偏移
def get_dev(u, t, beta, user_meanday):
    mean_day = user_meanday[u]
    deviation = t - mean_day
    ans = 0 if deviation == 0 else (deviation) / math.fabs(deviation) * \
        math.fabs(deviation) ** beta
    return nd.array([ans])


# 求一个日期对应的bin的编号
def get_bin_index(t, bin_cnt, day_cnt):
    bin_size = math.ceil(day_cnt / bin_cnt)
    return t // bin_size


def get_vector(i, cnt):
    i = int(i)
    vec = nd.zeros((1, cnt))
    vec[0][i] = 1
    return vec


class Trainer():
    # 记录所有数据
    def __init__(self, items_of_user, rating_cnt, test_items_of_user,
                 test_rating_cnt, user_meanday, item_cnt, user_cnt, day_cnt,
                 average_rating, bin_size, beta):
        self.items_of_user = items_of_user
        self.rating_cnt = rating_cnt
        self.test_items_of_user = test_items_of_user
        self.test_rating_cnt = test_rating_cnt
        self.user_cnt = user_cnt
        self.user_meanday = user_meanday
        self.item_cnt = item_cnt
        self.user_cnt = user_cnt
        self.day_cnt = day_cnt
        self.average_rating = average_rating
        self.bin_size = bin_size
        self.beta = beta
        self.R = {}
        self.test_R = {}
        # 对每个用户统计他评过的商品
        for user in self.items_of_user.keys():
            self.R[user] = (nd.zeros((1, self.item_cnt)))
            self.test_R[user] = (nd.zeros((1, self.item_cnt)))
            for item in self.items_of_user[user]:
                self.R[user][0][item[0]] = 1
                self.test_R[user][0][item[0]] = 1
        for user in self.test_items_of_user.keys():
            for item in self.items_of_user[user]:
                self.test_R[user][0][item[0]] = 1
        # 将数据整理到一个序列中
        self.random_data = []
        for user in items_of_user.keys():
            for item in items_of_user[user]:
                self.random_data.append((user,) + item)

        # 定义模型以及训练器
        self.timeSVDpp = TimeSVDpp(items_of_user, user_meanday,
                                   item_cnt, user_cnt, day_cnt, average_rating)
        self.timeSVDpp.initialize()
        self.timeSVDpp.hybridize()

    # 测试模型
    def test(self):
        test_total_loss = 0
        tested_cnt = 0
        # 遍历测试集
        for user in self.test_items_of_user.keys():
            u = get_vector(user, self.user_cnt)
            y_i = self.test_R[user]
            for item in self.test_items_of_user[user]:
                i = get_vector(item[0], self.item_cnt)
                t = get_vector(item[2], self.day_cnt)
                bint = get_vector(get_bin_index(item[2], self.bin_size, self.day_cnt),
                                  self.bin_size)
                r_hat = self.timeSVDpp(u, i, t, y_i,
                                       get_dev(user, item[2], self.beta,
                                               self.user_meanday), bint)
                loss = (r_hat - item[1]) ** 2
                test_total_loss += loss
                tested_cnt += 1
                # 输出当前进度
                print('Testing, tested percent:',
                      tested_cnt / self.test_rating_cnt, '. \tLoss =',
                      math.sqrt((test_total_loss / tested_cnt)[0].asscalar()),
                      end='\r')
        # 输出测试结果
        print('Test finished, Loss =',
              math.sqrt(test_total_loss[0].asscalar()) / self.test_rating_cnt)

    # 训练模型
    def train(self, epoch_cnt, learning_method, learning_params, is_random=True):
        # 定义训练器
        trainer = gluon.Trainer(self.timeSVDpp.collect_params(),
                                learning_method, learning_params)

        # 训练过程
        for epoch in range(epoch_cnt):
            total_loss = 0
            trained_cnt = 0
            # 遍历所有用户
            if is_random is False:
                for user in self.items_of_user.keys():
                    u = get_vector(user, self.user_cnt)
                    y_i = self.R[user]
                    # 针对每个商品进行迭代
                    for item in self.items_of_user[user]:
                        i = get_vector(item[0], self.item_cnt)
                        t = get_vector(item[2], self.day_cnt)
                        bint = get_vector(get_bin_index(item[2], self.bin_size, self.day_cnt),
                                          self.bin_size)
                        with mxnet.autograd.record():
                            r_hat = self.timeSVDpp(u, i, t, y_i,
                                                   get_dev(user, item[2], self.beta,
                                                           self.user_meanday), bint)
                            loss = (r_hat - item[1]) ** 2
                        loss.backward()
                        trainer.step(1)
                        total_loss += loss
                        trained_cnt += 1
                        # 输出当前进度
                        print('Epoch', epoch, 'training, ',
                              trained_cnt / self.rating_cnt, 'trained. \tLoss =',
                              (total_loss / trained_cnt)[0].asscalar(), end='\r')
            else:
                random.shuffle(self.random_data)
                # 针对每个评分项进行迭代
                for rating in self.random_data:
                    user = rating[0]
                    u = get_vector(user, self.user_cnt)
                    y_i = self.R[user]
                    i = get_vector(rating[1], self.item_cnt)
                    t = get_vector(rating[3], self.day_cnt)
                    bint = get_vector(get_bin_index(rating[3], self.bin_size, self.day_cnt),
                                      self.bin_size)
                    with mxnet.autograd.record():
                        r_hat = self.timeSVDpp(u, i, t, y_i,
                                               get_dev(user, rating[3], self.beta,
                                                       self.user_meanday), bint)
                        loss = (r_hat - rating[2]) ** 2
                    loss.backward()
                    trainer.step(1)
                    total_loss += loss
                    trained_cnt += 1
                    # 输出当前进度
                    print('Epoch', epoch, 'training, ',
                          trained_cnt / self.rating_cnt, 'trained. \tLoss =',
                          (total_loss / trained_cnt)[0].asscalar(), end='\r')
            # 输出结果
            print('Epoch', epoch, 'finished, Loss =',
                  total_loss[0].asscalar() / self.rating_cnt)
            # 测试效果
            self.test()
