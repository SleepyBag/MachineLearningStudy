import pdb
import sys
import numpy as np
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import AlgoBase
from surprise.model_selection import cross_validate
from surprise import PredictionImpossible
from matplotlib import pyplot as plt
import math

data = Dataset.load_builtin('ml-100k')

algo = SVD(n_epochs=60)
print('SVD训练结果')
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

algo = SVDpp(n_epochs=60)
print('SVD++训练结果')
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


class kernelSVD(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user biases
        bu = np.zeros(trainset.n_users)
        # item biases
        bi = np.zeros(trainset.n_items)
        # user factors
        pu = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_users, self.n_factors))
        # item factors
        qi = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_items, self.n_factors))

        # int u, i, f
        # double r, err, dot, puf, qif
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        if not self.biased:
            global_mean = 0

        err_log = []

        bool_continue = 'y'
        while bool_continue != 'n':

            for current_epoch in range(self.n_epochs):
                if self.verbose:
                    print("Processing epoch {}".format(current_epoch))
                total_err = 0
                for u, i, r in trainset.all_ratings():

                    # compute current error
                    dis = 0  # p与q的距离
                    for f in range(self.n_factors):
                        dis += (qi[i, f] - pu[u, f]) ** 2
                    kernel = math.exp(-dis ** 2 / 2)  # 求核函数
                    err = r - (global_mean + bu[u] + bi[i] + kernel)
                    total_err += err

                    # update biases
                    if self.biased:
                        bu[u] += lr_bu * (err - reg_bu * bu[u])
                        bi[i] += lr_bi * (err - reg_bi * bi[i])

                    # update factors
                    for f in range(self.n_factors):
                        puf = pu[u, f]
                        qif = qi[i, f]
                        # d_kernel_pu = math.exp()
                        pu[u, f] += lr_pu * \
                            (err * kernel * (qif - puf) - reg_pu * puf)
                        qi[i, f] += lr_qi * \
                            (err * kernel * (puf - qif) - reg_qi * qif)

                total_err /= trainset.n_ratings
                err_log.append(total_err)
                sys.stderr.write('Epoch ' + str(current_epoch) +
                                 ' : ' + str(total_err) + '\n')

            # for i, err in enumerate(err_log):
            #     plt.plot(i, err, 'r.')
            # plt.show()
            # sys.stderr.write('继续训练吗?(y/n)')
            bool_continue = 'n'

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unkown.')

        return est


algo = kernelSVD(n_epochs=60)
print('Kernel SVD训练结果')
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
