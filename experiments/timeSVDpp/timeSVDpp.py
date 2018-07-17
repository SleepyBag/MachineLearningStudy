from mxnet import gluon
from load_movie_data import loadMovieData
from mxnet import ndarray as nd
import math

data_loader = loadMovieData()
userItems, nUsers, nItems, nDays, minTimestamp = data_loader.main()

# 计算总平均分数
average_rating = 0
rating_cnt = 0
for user in userItems.keys():
    items = userItems[user]
    for item in items:
        average_rating += item[1]
        rating_cnt += 1
average_rating /= rating_cnt

# 为每一个用户计算他所有评分日的平均数
user_meanday = {}
for user in userItems.keys():
    items = userItems[user]
    meanday = 0
    for item in items:
        meanday += item[2] / len(items)
    user_meanday[user] = meanday


class TimeSVDpp(gluon.nn.Block):

    def __init__(self, factor_cnt, items_of_user, user_meanday,
                 item_cnt, user_cnt, day_cnt, average_rating, bin_size=30, beta=.4, ** kwargs):
        super(TimeSVDpp, self).__init__(**kwargs)

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
        bin_cnt = (day_cnt + 1) // bin_size

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

    # 求某一个日期对某用户的平均评分日期的偏移
    def _get_dev(self, u, t):
        mean_day = user_meanday[u]
        deviation = t - mean_day
        return (deviation) / math.abs(deviation) * \
            math.abs(deviation) ** self.beta

    # 求一个日期对应的bin的编号
    def _get_bin_index(self, t):
        return t // self.bin_size

    def forward(self, u, i, t, r):
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

        dev = self._get_dev(u, t)

        b_item = b_item_long[i] + b_item_short[i,
                                               self._get_bin_index(t)]   # 物品偏移
        b_user = b_user_long[u] + alpha_bias[u] * \
            dev + b_user_short[u, t]  # 用户偏移
        p = p_long[u] + alpha_preference[u] * \
            dev + p_short[u, t]           # 用户偏好

        # 求用户评分过的商品所蕴含的用户偏好
        sum_y = nd.zeros(shape=(self.factor_cnt))
        for item in self.items_of_user[u]:
            item = item[0]
            sum_y += y[item]
        sum_y /= math.sqrt(len(self.items_of_user[u]))

        # 预测评分
        r_hat = self.mu + b_item + b_user + \
            nd.dot(q.reshape((1, -1)), p.reshape(1, -1) + sum_y)
        return r_hat
