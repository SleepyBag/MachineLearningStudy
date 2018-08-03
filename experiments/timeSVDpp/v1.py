from mxnet import gluon
import mxnet
from mxnet import ndarray as nd
import math
import random
from tqdm import tqdm


class TimeSVDpp(gluon.nn.HybridBlock):

    def __init__(self, item_cnt, user_cnt, day_cnt, average_rating,
                 factor_cnt, bin_cnt, beta, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

        # 复制数据到对象中
        self.factor_cnt = factor_cnt
        self.item_cnt = item_cnt
        self.user_cnt = user_cnt
        self.day_cnt = day_cnt
        self.mu = average_rating

        self.beta = beta
        self.bin_cnt = bin_cnt

        with self.name_scope():
            # 定义MLP块
            self.mlp1_W1 = self.params.get('mlp1_W1',
                                           shape=(2 * factor_cnt, factor_cnt))
            self.mlp1_b1 = self.params.get('mlp1_b1', shape=(1, factor_cnt))
            self.mlp1_W2 = self.params.get('mlp1_W2',
                                           shape=(factor_cnt, factor_cnt // 2))
            self.mlp1_b2 = self.params.get('mlp1_b2', shape=(1, factor_cnt // 2))
            self.mlp1_W3 = self.params.get('mlp1_W3', shape=(factor_cnt // 2, 1))
            self.mlp1_b3 = self.params.get('mlp1_b3', shape=(1, 1))
            self.mlp1_dropout = gluon.nn.Dropout(.5)

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
            self.p_long = self.params.get('p_long', shape=(user_cnt, factor_cnt))
            self.alpha_preference = self.params.get(
                'alpha_preference', shape=(user_cnt, factor_cnt))
            self.p_short = self.params.get(
                'p_short', shape=(user_cnt, day_cnt, factor_cnt))

    def hybrid_forward(self, F, u, i, t, R_u, dev, bint, q, y, b_item_long,
                       b_item_short, b_user_long, alpha_bias, b_user_short,
                       p_long, alpha_preference, p_short,
                       mlp1_W1, mlp1_b1, mlp1_W2, mlp1_b2, mlp1_W3, mlp1_b3):
        # 根据下标取数据
        b_item_long_i = F.dot(i, b_item_long).reshape((1, 1))
        b_item_short_i_bint = F.dot(bint, F.transpose(F.dot(i, b_item_short)))
        b_user_long_u = F.dot(u, b_user_long).reshape((1, 1))
        alpha_bias_u = F.dot(u, alpha_bias).reshape((1, 1))
        b_user_short_u_t = F.dot(t, F.transpose(
            F.dot(u, b_user_short))).reshape((1, 1))
        p_long_u = F.dot(u, p_long)
        alpha_preference_u = F.dot(u, alpha_preference)
        p_short_u = F.dot(t, F.dot(u, p_short).reshape(
            (self.day_cnt, self.factor_cnt)))

        b_item = b_item_long_i + b_item_short_i_bint                   # 商品偏移
        b_user = alpha_bias_u * dev.reshape((1, 1)) + \
            b_user_short_u_t + b_user_long_u                           # 用户偏移
        p = p_long_u + p_short_u + alpha_preference_u * \
            F.broadcast_to(dev.reshape((1, 1)), (1, self.factor_cnt))  # 用户偏好
        sum_y = F.dot(R_u, y) / F.broadcast_to(F.sqrt(F.sum(R_u).reshape((1, 1))),
                                               (1, self.factor_cnt))   # 商品影响的用户偏好
        q_i = F.dot(i, q)                                              # 商品属性

        # MLP输出
        mlp = F.concat(q_i, p, dim=1)
        mlp = F.dot(mlp, mlp1_W1) + mlp1_b1
        mlp = F.relu(mlp)
        mlp = F.dot(mlp, mlp1_W2) + mlp1_b2
        mlp = F.relu(mlp)
        mlp = self.mlp1_dropout(mlp)
        mlp = F.dot(mlp, mlp1_W3) + mlp1_b3

        # 预测评分
        r_hat = self.mu + b_item + b_user + mlp +\
            F.dot(q_i, F.transpose(p) + F.transpose(sum_y))
        # 计算正则项
        reg = b_item_long_i ** 2 + b_item_short_i_bint ** 2 + b_user_long_u ** 2 + \
            alpha_bias_u ** 2 + b_user_short_u_t ** 2 + F.sum(p_long_u ** 2).reshape((1, 1)) + \
            F.sum(alpha_preference_u ** 2).reshape((1, 1)) + F.sum(p_short_u ** 2).reshape((1, 1)) + \
            F.sum(q_i ** 2).reshape((1, 1)) + F.sum(F.dot(R_u, y ** 2)).reshape((1, 1)) + \
            F.sum(mlp1_W1 ** 2).reshape((1, 1)) + F.sum(mlp1_b1 ** 2).reshape((1, 1)) + \
            F.sum(mlp1_W2 ** 2).reshape((1, 1)) + F.sum(mlp1_b2 ** 2).reshape((1, 1))

        return r_hat, reg


class Trainer():
    # 记录所有数据
    def __init__(self, items_of_user, rating_cnt, test_items_of_user,
                 test_rating_cnt, user_meanday, item_cnt, user_cnt, day_cnt,
                 average_rating, factor_cnt, bin_cnt, beta):
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
        self.bin_cnt = bin_cnt
        self.beta = beta
        self.R = {}
        # self.test_R = {}
        # 对每个用户统计他评过的商品
        for user in self.items_of_user.keys():
            self.R[user] = (nd.zeros((1, self.item_cnt)))
            # self.test_R[user] = (nd.zeros((1, self.item_cnt)))
            for item in self.items_of_user[user]:
                self.R[user][0][item[0]] = 1
                # self.test_R[user][0][item[0]] = 1
        # for user in self.test_items_of_user.keys():
        #     for item in self.items_of_user[user]:
        #         self.test_R[user][0][item[0]] = 1
        # 将数据整理到一个序列中
        self.random_data = []
        for user in items_of_user.keys():
            for item in items_of_user[user]:
                self.random_data.append((user,) + item)

        # 定义模型以及训练器
        self.timeSVDpp = TimeSVDpp(item_cnt, user_cnt, day_cnt, average_rating,
                                   factor_cnt, bin_cnt, beta)
        self.timeSVDpp.initialize()
        self.timeSVDpp.hybridize()

    # 求一个日期对应的bin的编号
    def get_bin_index(self, t, bin_cnt, day_cnt):
        bin_size = math.ceil(day_cnt / bin_cnt)
        return t // bin_size

    def get_vector(self, i, cnt):
        i = int(i)
        vec = nd.zeros((1, cnt))
        vec[0][i] = 1
        return vec

    # 求某一个日期对某用户的平均评分日期的偏移
    def get_dev(self, u, t, beta, user_meanday):
        mean_day = user_meanday[u]
        deviation = t - mean_day
        ans = 0 if deviation == 0 else deviation / math.fabs(deviation) * \
            math.fabs(deviation) ** beta
        return nd.array([ans])

    def get_vectors(self, user, item, time):
        u = self.get_vector(user, self.user_cnt)
        R_u = self.R[user]
        i = self.get_vector(item, self.item_cnt)
        t = self.get_vector(time, self.day_cnt)
        bint = self.get_vector(self.get_bin_index(time, self.bin_cnt,
                                                  self.day_cnt), self.bin_cnt)
        dev = self.get_dev(user, time, self.beta, self.user_meanday)
        return u, R_u, i, t, bint, dev

    # 测试模型
    def test(self):
        test_total_loss = 0
        tested_cnt = 0
        # 遍历测试集
        for user in self.test_items_of_user.keys():
            # R_u = self.test_R[user]
            for item in self.test_items_of_user[user]:
                u, R_u, i, t, bint, dev = self.get_vectors(user, item[0], item[2])
                r_hat, reg = self.timeSVDpp(u, i, t, R_u, dev, bint)
                loss = (r_hat - item[1]) ** 2
                test_total_loss += loss
                tested_cnt += 1
        # 输出测试结果
        print('Test finished, Loss =',
              math.sqrt(test_total_loss[0].asscalar() / self.test_rating_cnt))

    # 训练模型
    def train(self, epoch_cnt, learning_method, learning_params, is_random=True):
        # 定义训练器
        trainer_learning_params = learning_params
        trainer_learning_params['wd'] = 0
        trainer = gluon.Trainer(self.timeSVDpp.collect_params(),
                                learning_method, trainer_learning_params)

        # 训练过程
        for epoch in range(epoch_cnt):
            total_loss = 0
            trained_cnt = 0
            cur_loss = 0
            if is_random is True:
                random.shuffle(self.random_data)
            # 针对每个评分项进行迭代
            pbar = tqdm(self.random_data)
            for rating in pbar:
                u, R_u, i, t, bint, dev = self.get_vectors(rating[0], rating[1],
                                                           rating[3])
                with mxnet.autograd.record():
                    r_hat, reg = self.timeSVDpp(u, i, t, R_u, dev, bint)
                    loss = (r_hat - rating[2]) ** 2 + learning_params['wd'] * reg
                loss.backward()
                trainer.step(1)
                total_loss += loss
                trained_cnt += 1
                if trained_cnt % 5000 == 0:
                    cur_loss = (total_loss / trained_cnt)[0].asscalar()
                pbar.set_description('Loss=%6f' % cur_loss)
            # 输出结果
            print('Epoch', epoch, 'finished, Loss =',
                  total_loss[0].asscalar() / self.rating_cnt)
            # 测试效果
            self.test()
