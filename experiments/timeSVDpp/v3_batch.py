from mxnet import gluon
import mxnet
from mxnet import ndarray as nd
import math
import random
from tqdm import tqdm


class TimeSVDpp(gluon.nn.HybridBlock):

    def __init__(self, item_cnt, user_cnt, day_cnt, average_rating,
                 factor_cnt, bin_cnt, beta, batch_size, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

        # 复制数据到对象中
        self.factor_cnt = factor_cnt
        self.item_cnt = item_cnt
        self.user_cnt = user_cnt
        self.day_cnt = day_cnt
        self.mu = average_rating
        self.beta = beta
        self.bin_cnt = bin_cnt
        self.batch_size = batch_size

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
            self.q_mlp = self.params.get('q_mlp', shape=(item_cnt, factor_cnt))
            self.y = self.params.get('y', shape=(item_cnt, factor_cnt))
            self.y_mlp = self.params.get('y_mlp', shape=(item_cnt, factor_cnt))

            # 设定学习参数p_long,alpha_preference与p_short,即用户的偏好,用户偏好的长期变化系数,用户爱好的单日变化
            self.p_long = self.params.get('p_long', shape=(user_cnt, factor_cnt))
            self.p_long_mlp = self.params.get('p_long_mlp',
                                              shape=(user_cnt, factor_cnt))
            self.alpha_preference = self.params.get(
                'alpha_preference', shape=(user_cnt, factor_cnt))
            self.alpha_preference_mlp = self.params.get(
                'alpha_preference_mlp', shape=(user_cnt, factor_cnt))
            self.p_short = self.params.get(
                'p_short', shape=(user_cnt, day_cnt, factor_cnt))
            self.p_short_mlp = self.params.get(
                'p_short_mlp', shape=(user_cnt, day_cnt, factor_cnt))

            # 设定学习参数b_item_long与b_item_short,即物品的长期偏移与临时偏移
            self.b_item_long = self.params.get('b_item_long', shape=(item_cnt))
            self.b_item_short = self.params.get(
                'b_item_short', shape=(item_cnt, bin_cnt))

            # 设定学习参数b_user_long,alpha_bias与b_user_short,即用户的长期偏移,用户偏移的长期变化系数,和用户的单日偏移
            self.b_user_long = self.params.get('b_user_long', shape=(user_cnt))
            self.alpha_bias = self.params.get('alpha_bias', shape=(user_cnt))
            self.b_user_short = self.params.get(
                'b_user_short', shape=(user_cnt, day_cnt))

    def hybrid_forward(self, F, u, i, t, R_u, dev, bint, q, y, b_item_long,
                       b_item_short, b_user_long, alpha_bias, b_user_short,
                       p_long, alpha_preference, p_short, p_long_mlp,
                       p_short_mlp, alpha_preference_mlp, q_mlp, y_mlp,
                       mlp1_W1, mlp1_b1, mlp1_W2, mlp1_b2, mlp1_W3, mlp1_b3):
        # 根据下标取数据
        # 偏移部分
        b_item_long_i = F.dot(i, b_item_long).reshape((self.batch_size, 1))
        b_item_short_i_bint = \
            F.sum(bint * F.dot(i, b_item_short), axis=1).reshape((self.batch_size, 1))
        b_user_long_u = F.dot(u, b_user_long).reshape((self.batch_size, 1))
        alpha_bias_u = F.dot(u, alpha_bias).reshape((self.batch_size, 1))
        b_user_short_u_t = \
            F.sum(t * F.dot(u, b_user_short), axis=1).reshape((self.batch_size, 1))
        # 偏好部分
        p_long_u = F.dot(u, p_long)
        alpha_preference_u = F.dot(u, alpha_preference)
        p_short_u = F.sum(F.broadcast_axis(
            t.reshape((self.batch_size, self.day_cnt, 1)), axis=2, size=self.factor_cnt) *
            F.dot(u, p_short), axis=1)
        p_long_u_mlp = F.dot(u, p_long_mlp)
        alpha_preference_u_mlp = F.dot(u, alpha_preference_mlp)
        p_short_u_mlp = F.sum(F.broadcast_axis(
            t.reshape((self.batch_size, self.day_cnt, 1)), axis=2, size=self.factor_cnt) *
            F.dot(u, p_short_mlp), axis=1)
        q_i_mlp = F.dot(i, q_mlp)

        b_item = b_item_long_i + b_item_short_i_bint                           # 商品偏移
        b_user = alpha_bias_u * dev + b_user_short_u_t + b_user_long_u         # 用户偏移
        p_u = p_long_u + p_short_u + F.broadcast_mul(alpha_preference_u, dev)  # 用户偏好
        p_u_mlp = p_long_u_mlp + p_short_u_mlp + F.broadcast_mul(alpha_preference_u_mlp, dev)  # 用户偏好
        sum_y = F.broadcast_div(F.dot(R_u, y), F.sqrt(F.sum(R_u, axis=1))
                                .reshape((self.batch_size, 1)))                # 商品影响的用户偏好
        sum_y_mlp = F.broadcast_div(F.dot(R_u, y_mlp), F.sqrt(F.sum(R_u, axis=1))
                                    .reshape((self.batch_size, 1)))                # 商品影响的用户偏好
        q_i = F.dot(i, q)                                                      # 商品属性

        # MLP输出
        mlp = F.concat(p_u_mlp + sum_y_mlp, q_i_mlp, dim=1)
        mlp = F.broadcast_add(F.dot(mlp, mlp1_W1), mlp1_b1)
        mlp = F.relu(mlp)
        mlp = F.broadcast_add(F.dot(mlp, mlp1_W2), mlp1_b2)
        mlp = F.relu(mlp)
        mlp = self.mlp1_dropout(mlp)
        mlp = F.broadcast_add(F.dot(mlp, mlp1_W3), mlp1_b3)

        # 点乘输出
        dot = F.sum(q_i * (p_u + sum_y), axis=1).reshape((self.batch_size, 1))
        # # MLP与点乘分别所占的比例
        # mlp_rate = (F.ones((1, 1)) - alpha)
        # dot_rate = alpha

        # 预测评分
        # r_hat = self.mu + b_item + b_user + mlp_rate * mlp + dot_rate * dot
        r_hat = self.mu + b_item + b_user + mlp + dot

        # 计算正则项
        # MLP部分
        mlp_net_reg = F.sum(mlp1_W1 ** 2) + F.sum(mlp1_b1 ** 2) + \
            F.sum(mlp1_W2 ** 2) + F.sum(mlp1_b2 ** 2) + \
            F.sum(mlp1_W3 ** 2) + F.sum(mlp1_b3 ** 2)
        mlp_net_reg = mlp_net_reg.reshape((1, 1))
        mlp_reg = F.sum(p_long_u_mlp ** 2, axis=1) + \
            F.sum(alpha_preference_u_mlp ** 2, axis=1) + F.sum(p_short_u_mlp ** 2, axis=1) + \
            F.sum(q_i_mlp ** 2, axis=1) + F.sum(F.dot(R_u, y_mlp ** 2), axis=1)
        mlp_reg = F.broadcast_add(mlp_net_reg, mlp_reg)
        # 点乘部分
        dot_reg = F.sum(p_long_u ** 2, axis=1) + \
            F.sum(alpha_preference_u ** 2, axis=1) + F.sum(p_short_u ** 2, axis=1) + \
            F.sum(q_i ** 2, axis=1) + F.sum(F.dot(R_u, y ** 2), axis=1)
        dot_reg = dot_reg.reshape((self.batch_size, 1))
        # 偏差部分
        bias_reg = b_item_long_i ** 2 + b_item_short_i_bint ** 2 + \
            b_user_long_u ** 2 + alpha_bias_u ** 2 + b_user_short_u_t ** 2
        # 汇总
        reg = bias_reg + mlp_reg + dot_reg
        # reg = reg * self.batch_size

        return r_hat, reg


class Trainer():
    def get_vector(self, i, cnt):
        i = int(i)
        vec = nd.zeros((1, cnt))
        vec[0][i] = 1
        return vec

    # 记录所有数据
    def __init__(self, items_of_user, rating_cnt, test_items_of_user,
                 test_rating_cnt, user_meanday, item_cnt, user_cnt, day_cnt,
                 average_rating, factor_cnt, bin_cnt, beta, batch_size=10):
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
        self.batch_size = batch_size
        self.R = {}

        # 获取所有向量
        self.user_vec = []
        self.item_vec = []
        self.time_vec = []
        self.bin_vec = []
        for user in range(user_cnt):
            self.user_vec.append(self.get_vector(user, user_cnt))
        for item in range(item_cnt):
            self.item_vec.append(self.get_vector(item, item_cnt))
        for time in range(day_cnt):
            self.time_vec.append(self.get_vector(time, day_cnt))
        for b in range(bin_cnt):
            self.bin_vec.append(self.get_vector(b, bin_cnt))
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
        self.test_data = []
        for user in test_items_of_user.keys():
            for item in test_items_of_user[user]:
                self.test_data.append((user,) + item)

        # 定义模型以及训练器
        self.timeSVDpp = TimeSVDpp(item_cnt, user_cnt, day_cnt, average_rating,
                                   factor_cnt, bin_cnt, beta, batch_size)
        self.timeSVDpp.initialize()
        # self.timeSVDpp.hybridize()

    # 求一个日期对应的bin的编号
    def get_bin_index(self, t, bin_cnt, day_cnt):
        bin_size = math.ceil(day_cnt / bin_cnt)
        return t // bin_size

    # 求某一个日期对某用户的平均评分日期的偏移
    def get_dev(self, u, t, beta, user_meanday):
        mean_day = user_meanday[u]
        deviation = t - mean_day
        ans = 0 if deviation == 0 else deviation / math.fabs(deviation) * \
            math.fabs(deviation) ** beta
        return nd.array([ans])

    def get_vectors(self, user, item, time):
        u = self.user_vec[user]
        R_u = self.R[user]
        i = self.item_vec[item]
        t = self.time_vec[time]
        bint = self.bin_vec[self.get_bin_index(time, self.bin_cnt, self.day_cnt)]
        dev = self.get_dev(user, time, self.beta, self.user_meanday)
        return u, R_u, i, t, bint, dev

    # 测试模型
    def test(self):
        test_total_loss = 0
        tested_cnt = 0
        data = self.batchify(self.test_data, self.batch_size, is_random=False)
        pbar = tqdm(data)
        # 遍历测试集
        for rating in pbar:
            tested_cnt += self.batch_size
            # R_u = self.test_R[user]
            u, R_u, i, t, bint, dev, r = rating
            r_hat, reg = self.timeSVDpp(u, i, t, R_u, dev, bint)
            loss = (r_hat - r) ** 2
            test_total_loss += nd.sum(loss)
            cur_loss = (test_total_loss / tested_cnt)[0].asscalar()
            pbar.set_description('Loss=%.6f' % cur_loss)
        # 输出测试结果
        print('Test finished, Loss =',
              math.sqrt(test_total_loss[0].asscalar() / self.test_rating_cnt))

    def batchify(self, data, batch_size, is_random):
        if is_random is True:
            random.shuffle(data)
        batched = []
        batched_cnt = 0
        for rating in data:
            if batched_cnt == 0:
                bu, bR_u, bi, bt, bbint, bdev, br = (tuple(),) * 7
            u, R_u, i, t, bint, dev = self.get_vectors(rating[0], rating[1],
                                                       rating[3])
            bu = bu + (u,)
            bR_u = bR_u + (R_u,)
            bi = bi + (i,)
            bt = bt + (t,)
            bbint = bbint + (bint,)
            bdev = bdev + (dev,)
            br = br + (nd.array([[rating[2]]]),)
            batched_cnt += 1
            if batched_cnt % batch_size == 0:
                bu = nd.concat(*bu, dim=0)
                bR_u = nd.concat(*bR_u, dim=0)
                bi = nd.concat(*bi, dim=0)
                bt = nd.concat(*bt, dim=0)
                bbint = nd.concat(*bbint, dim=0)
                bdev = nd.concat(*bdev, dim=0).reshape((batch_size, 1))
                br = nd.concat(*br, dim=0).reshape((batch_size, 1))
                batched.append((bu, bR_u, bi, bt, bbint, bdev, br))
                batched_cnt = 0
        return batched

    # 训练模型
    def train(self, epoch_cnt, learning_method, learning_params, verbose,
              is_random=True):
        # 定义训练器
        wd = learning_params['wd']
        learning_params['wd'] = 0
        trainer = gluon.Trainer(self.timeSVDpp.collect_params(),
                                learning_method, learning_params)
        # alpha = nd.array([alpha]).reshape((1, 1))

        # 训练过程
        for epoch in range(epoch_cnt):
            total_loss = 0
            trained_cnt = 0
            data = self.batchify(self.random_data, self.batch_size, is_random)
            # 针对每个评分项进行迭代
            pbar = tqdm(data)
            for rating in pbar:
                trained_cnt += self.batch_size
                u, R_u, i, t, bint, dev, r = rating
                with mxnet.autograd.record():
                    r_hat, reg = self.timeSVDpp(u, i, t, R_u, dev, bint)
                    loss = nd.sum((r_hat - r) ** 2 + wd * reg)
                loss.backward()
                trainer.step(self.batch_size)

                total_loss += nd.sum(loss)
                if trained_cnt % 500 == 0:
                    cur_loss = (total_loss / trained_cnt)[0].asscalar()
                    pbar.set_description('Loss=%.6f' % cur_loss)
            # # 输出结果
            # print('Epoch', epoch, 'finished, Loss =',
            #       total_loss[0].asscalar() / self.rating_cnt)
            # 测试效果
            if epoch >= verbose:
                self.test()
