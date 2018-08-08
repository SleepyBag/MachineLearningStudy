import mxnet
from mxnet import gluon
from mxnet import ndarray as nd
import math
from tqdm import tqdm
from mxnet.gluon import data as gdata


class Trainer():
    def get_vector(self, i, cnt):
        i = int(i)
        vec = nd.zeros((1, cnt))
        vec[0][i] = 1
        return vec

    # 记录所有数据
    def __init__(self, items_of_user, rating_cnt, test_items_of_user,
                 test_rating_cnt, user_meanday, item_cnt, user_cnt, day_cnt,
                 average_rating, factor_cnt, bin_cnt, beta, batch_size):
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
        self.train_data = []
        for user in items_of_user.keys():
            for item in items_of_user[user]:
                self.train_data.append((user,) + item)
        self.test_data = []
        for user in test_items_of_user.keys():
            for item in test_items_of_user[user]:
                self.test_data.append((user,) + item)
        self.train_data = self.batchify(self.train_data)
        self.test_data = self.batchify(self.test_data)
        self.train_dataset = gdata.ArrayDataset(*self.train_data)
        self.test_dataset = gdata.ArrayDataset(*self.test_data)

        # # 定义模型以及训练器
        # self.model = model(item_cnt, user_cnt, day_cnt, average_rating,
        #                    factor_cnt, bin_cnt, beta, batch_size)
        # self.model.initialize()
        # self.model.hybridize()

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
        return u, R_u, i, t, bint, dev.reshape((1, 1))

    # 测试模型
    def test(self, progress, model):
        test_total_loss = 0
        tested_cnt = 0
        data = gdata.DataLoader(self.test_dataset, batch_size=self.batch_size)
        if progress is True:
            data = tqdm(data)
        # 遍历测试集
        for u, R_u, i, t, bint, dev, r in data:
            tested_cnt += self.batch_size
            # R_u = self.test_R[user]
            r_hat, reg = model(u, i, t, R_u, dev, bint)
            loss = (r_hat - r) ** 2
            test_total_loss += nd.sum(loss).asscalar()
            cur_loss = test_total_loss / tested_cnt
            if progress is True:
                data.set_description('MSE=%.6f' % cur_loss)
        # 输出测试结果
        print('Test finished, RMSE =',
              math.sqrt(test_total_loss / self.test_rating_cnt))

    def batchify(self, data):
        # 逐个获取向量化的数据
        bu, bR_u, bi, bt, bbint, bdev, br = ([], [], [], [], [], [], [],)
        for rating in data:
            u, R_u, i, t, bint, dev = self.get_vectors(rating[0], rating[1],
                                                       rating[3])
            bu.append(u)
            bR_u.append(R_u)
            bi.append(i)
            bt.append(t)
            bbint.append(bint)
            bdev.append(dev)
            br.append(nd.array([[rating[2]]]))
        bu = nd.concat(*bu, dim=0)
        bR_u = nd.concat(*bR_u, dim=0)
        bi = nd.concat(*bi, dim=0)
        bt = nd.concat(*bt, dim=0)
        bbint = nd.concat(*bbint, dim=0)
        bdev = nd.concat(*bdev, dim=0)
        br = nd.concat(*br, dim=0)
        return bu, bR_u, bi, bt, bbint, bdev, br

    # 训练模型
    def train(self, epoch_cnt, learning_method, learning_params, verbose, model,
              is_random=True, progress=False):
        # 定义训练器
        wd = learning_params['wd']
        learning_params['wd'] = 0
        dense_trainer = gluon.Trainer(model.collect_params(select='.*mlp[0-9]'),
                                      learning_method, learning_params)
        svd_trainer = gluon.Trainer(model.collect_params(select='.*_(q|y|p|alpha|b)(_|$)'),
                                    learning_method, learning_params)
        # alpha = nd.array([alpha]).reshape((1, 1))

        # 训练过程
        for epoch in range(epoch_cnt):
            total_loss = 0
            trained_cnt = 0
            data = gdata.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                    shuffle=is_random)
            # 针对每个评分项进行迭代
            if progress is True:
                data = tqdm(data)
            for u, R_u, i, t, bint, dev, r in data:
                trained_cnt += self.batch_size
                # 预测结果
                with mxnet.autograd.record():
                    r_hat, reg = model(u, i, t, R_u, dev, bint)
                    loss = (r_hat - r) ** 2 + wd * reg
                loss.backward()
                # 调整参数
                dense_trainer.step(self.batch_size)
                svd_trainer.step(math.sqrt(self.batch_size))

                total_loss += nd.sum(loss).asscalar()
                cur_loss = total_loss / trained_cnt
                if progress is True:
                    data.set_description('MSE=%.6f' % cur_loss)
            # # 输出结果
            # print('Epoch', epoch, 'finished, Loss =',
            #       total_loss[0].asscalar() / self.rating_cnt)
            # 测试效果
            if epoch >= verbose:
                self.test(progress, model)
