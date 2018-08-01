from load_movie_data import loadMovieData
import mxnet
import timeSVDpp_hybridize
import v1
import v2
import v3


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
        ans = int(ans)
        return ans
    else:
        return default


context = input('Choose a device(c for cpu, number for gpu(n)):')
if context == 'c':
    context = mxnet.cpu(0)
else:
    context = mxnet.gpu(int(context))

with mxnet.Context(context):
    model = input('Which model do you want to train?\n' +
                  '1. timeSVD++\t2. neuralTimeSVD++ v1\t3. neuralTimeSVD++ v2\t 4. neuralTimeSVD++ v3\n')
    bin_cnt = input_param('bin_cnt', 30)
    beta = input_param('beta', .4)
    factor_cnt = input_param('factor_cnt', 10)
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

    # timeSVD++
    if model == '1':
        timeSVDpp_trainer = timeSVDpp_hybridize. \
            Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                    user_meanday, nItems, nUsers, nDays, average_rating,
                    factor_cnt, bin_cnt, beta)
        timeSVDpp_trainer.train(epoch_cnt, learning_method, learning_params)
    # neuralTimeSVD++ v1
    if model == '2':
        v1_trainer = v1. \
            Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                    user_meanday, nItems, nUsers, nDays, average_rating,
                    factor_cnt, bin_cnt, beta)
        v1_trainer.train(epoch_cnt, learning_method, learning_params)
    # neuralTimeSVD++ v2
    elif model == '3':
        v2_trainer = v2. \
            Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                    user_meanday, nItems, nUsers, nDays, average_rating,
                    factor_cnt, bin_cnt, beta)
        v2_trainer.train(epoch_cnt, learning_method, learning_params, verbose)
    elif model == '4':
        v3_trainer = v3. \
            Trainer(userItems, rating_cnt, test_userItems, test_rating_cnt,
                    user_meanday, nItems, nUsers, nDays, average_rating,
                    factor_cnt, bin_cnt, beta)
        v3_trainer.train(epoch_cnt, learning_method, learning_params, verbose)
