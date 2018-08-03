from load_movie_data import loadMovieData
import mxnet
import timeSVDpp
import v1
import v2
import v3
import v2_batch
import math


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

nItems += 1
nUsers += 1


def input_param(prompt, default):
    ans = input(prompt + (' (default ' + str(default) + '):'))
    if ans != '':
        ans = float(ans)
        if math.floor(ans) == ans:
            ans = int(ans)
        return ans
    else:
        return default


context = input('Choose a device(c for cpu, number for gpu(n)):')
if context == 'c':
    context = mxnet.cpu(0)
else:
    context = mxnet.gpu(int(context))


def get_trainer():
    with mxnet.Context(context):
        model = input('Which model do you want to train?\n' +
                      '1. timeSVD++\t2. v1\t3. v2\t 4. v3 5. batchified v2\n')
        bin_cnt = input_param('bin_cnt', 30)
        beta = input_param('beta', .4)
        factor_cnt = input_param('factor_cnt', 10)
        batch_size = input_param('batch_size', 40)
        # timeSVD++
        if model == '1':
            trainer = timeSVDpp. \
                Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                        user_meanday, nItems, nUsers, nDays, average_rating,
                        factor_cnt, bin_cnt, beta)
        # neuralTimeSVD++ v1
        if model == '2':
            trainer = v1. \
                Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                        user_meanday, nItems, nUsers, nDays, average_rating,
                        factor_cnt, bin_cnt, beta)
        # neuralTimeSVD++ v2
        elif model == '3':
            trainer = v2. \
                Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                        user_meanday, nItems, nUsers, nDays, average_rating,
                        factor_cnt, bin_cnt, beta)
        elif model == '4':
            trainer = v3. \
                Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                        user_meanday, nItems, nUsers, nDays, average_rating,
                        factor_cnt, bin_cnt, beta)
        elif model == '5':
            trainer = v2_batch. \
                Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                        user_meanday, nItems, nUsers, nDays, average_rating,
                        factor_cnt, bin_cnt, beta, batch_size)
    return trainer


def train(trainer):
    with mxnet.Context(context):
        is_continue = 'y'
        while is_continue == 'y':
            learning_method = input('learning_method:(default sgd)')
            learning_params = {}
            if learning_method == '':
                learning_method = 'sgd'
            if learning_method == 'adam':
                learning_params['beta1'] = input_param('beta1', .9)
                learning_params['beta2'] = input_param('beta2', .999)
                learning_params['wd'] = input_param('wd', .01)
            elif learning_method == 'sgd':
                learning_params['learning_rate'] = input_param('learning_rate', .001)
                learning_params['wd'] = input_param('wd', .01)
            epoch_cnt = input_param('epoch_cnt', 20)
            verbose = input_param('verbose', 15)
            trainer.train(epoch_cnt, learning_method, learning_params, verbose)

            is_continue = input('continue?(y/n)')


trainer = get_trainer()
train(trainer)
