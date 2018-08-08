from mxnet import gluon


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
            # # 定义MLP块
            # self.mlp1_W1 = self.params.get('mlp1_W1',
            #                                shape=(2 * factor_cnt, factor_cnt))
            # self.mlp1_b1 = self.params.get('mlp1_b1', shape=(1, factor_cnt))
            # self.mlp1_W2 = self.params.get('mlp1_W2',
            #                                shape=(factor_cnt, factor_cnt // 2))
            # self.mlp1_b2 = self.params.get('mlp1_b2', shape=(1, factor_cnt // 2))
            # self.mlp1_W3 = self.params.get('mlp1_W3', shape=(factor_cnt // 2, 1))
            # self.mlp1_b3 = self.params.get('mlp1_b3', shape=(1, 1))
            # self.mlp1_dropout = gluon.nn.Dropout(.5)

            # # 设定MLP模块的学习参数p_mlp和q_mlp,分别代表用户和商品(不考虑时间变化)
            # self.p_mlp = self.params.get('p_mlp', shape=(user_cnt, factor_cnt))
            # self.q_mlp = self.params.get('q_mlp', shape=(item_cnt, factor_cnt))

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
                       p_long, alpha_preference, p_short):
        # 根据下标取数据
        b_item_long_i = F.dot(i, b_item_long).reshape((self.batch_size, 1))
        b_item_short_i_bint = \
            F.sum(bint * F.dot(i, b_item_short), axis=1).reshape((self.batch_size, 1))
        b_user_long_u = F.dot(u, b_user_long).reshape((self.batch_size, 1))
        alpha_bias_u = F.dot(u, alpha_bias).reshape((self.batch_size, 1))
        b_user_short_u_t = \
            F.sum(t * F.dot(u, b_user_short), axis=1).reshape((self.batch_size, 1))
        p_long_u = F.dot(u, p_long)
        alpha_preference_u = F.dot(u, alpha_preference)
        p_short_u = F.sum(F.broadcast_axis(
            t.reshape((self.batch_size, self.day_cnt, 1)), axis=2, size=self.factor_cnt) *
            F.dot(u, p_short), axis=1)
        # p_u_mlp = F.dot(u, p_mlp)
        # q_i_mlp = F.dot(i, q_mlp)

        b_item = b_item_long_i + b_item_short_i_bint                           # 商品偏移
        b_user = alpha_bias_u * dev + b_user_short_u_t + b_user_long_u         # 用户偏移
        p_u = p_long_u + p_short_u + F.broadcast_mul(alpha_preference_u, dev)  # 用户偏好
        sum_y = F.broadcast_div(F.dot(R_u, y), F.sqrt(F.sum(R_u, axis=1))
                                .reshape((self.batch_size, 1)))                # 商品影响的用户偏好
        q_i = F.dot(i, q)                                                      # 商品属性

        # # MLP输出
        # mlp = F.concat(p_u_mlp, q_i_mlp, dim=1)
        # mlp = F.broadcast_add(F.dot(mlp, mlp1_W1), mlp1_b1)
        # mlp = F.relu(mlp)
        # mlp = F.broadcast_add(F.dot(mlp, mlp1_W2), mlp1_b2)
        # mlp = F.relu(mlp)
        # mlp = self.mlp1_dropout(mlp)
        # mlp = F.broadcast_add(F.dot(mlp, mlp1_W3), mlp1_b3)

        # 点乘输出
        dot = F.sum(q_i * (p_u + sum_y), axis=1).reshape((self.batch_size, 1))
        # # MLP与点乘分别所占的比例
        # mlp_rate = (F.ones((1, 1)) - alpha)
        # dot_rate = alpha

        # 预测评分
        # r_hat = self.mu + b_item + b_user + mlp_rate * mlp + dot_rate * dot
        # r_hat = self.mu + b_item + b_user + mlp + dot
        r_hat = self.mu + b_item + b_user + dot

        # 计算正则项
        # mlp_reg = F.sum(mlp1_W1 ** 2) + F.sum(mlp1_b1 ** 2) + \
        #     F.sum(mlp1_W2 ** 2) + F.sum(mlp1_b2 ** 2) + \
        #     F.sum(mlp1_W3 ** 2) + F.sum(mlp1_b3 ** 2)
        # mlp_reg = mlp_reg.reshape((1, 1))
        # mlp_reg = F.broadcast_add(mlp_reg, (F.sum(p_u_mlp ** 2, axis=1) +
        #                                     F.sum(q_i_mlp ** 2, axis=1))
        #                           .reshape((self.batch_size, 1)))
        dot_reg = F.sum(p_long_u ** 2, axis=1) + \
            F.sum(alpha_preference_u ** 2, axis=1) + F.sum(p_short_u ** 2, axis=1) + \
            F.sum(q_i ** 2, axis=1) + F.sum(F.dot(R_u, y ** 2), axis=1)
        dot_reg = dot_reg.reshape((self.batch_size, 1))
        bias_reg = b_item_long_i ** 2 + b_item_short_i_bint ** 2 + \
            b_user_long_u ** 2 + alpha_bias_u ** 2 + b_user_short_u_t ** 2
        # reg = bias_reg + mlp_reg + dot_reg
        # reg = reg * self.batch_size
        reg = bias_reg + dot_reg

        return r_hat, reg
