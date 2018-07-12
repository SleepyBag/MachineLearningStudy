import pdb
import numpy as np
from math import log
from math import exp

learning_rate = .01

def p1(xhat, beta):
    e = np.exp(-np.dot(beta.T, xhat))[0][0]
    return 1 / (1 + e)


def iterate(X, Y, beta):
    import pdb
    # pdb.set_trace()
    grad = np.zeros(shape=beta.shape)
    grad2 = 0
    loss = 0
    for x, y in zip(X, Y):
        xhat = np.concatenate((np.array([x]).T, np.array([[1]])))
        y = y[0]
        grad += - xhat * (y - p1(xhat, beta))
        grad2 += np.dot(xhat, xhat.T) * p1(xhat, beta) * (1 - p1(xhat, beta))
        # pdb.set_trace()
        loss += log(1 + exp(-np.dot(beta.T, xhat))) + np.dot(beta.T, xhat) \
            - y * np.dot(beta.T, xhat)
    beta = beta - learning_rate * np.dot(np.linalg.inv(grad2), grad)
    return grad, grad2, beta, loss


def train(X, Y, beta):
    epoch = 20
    for i in range(epoch):
        grad, grad2, beta, loss = iterate(X, Y, beta)
        if i > 17:
            print('Epoch:', i, '\tloss =', loss)
    return beta


def test(X, Y, beta):
    correct_cnt = 0
    p = [[0] * output_num for i in range(Y[0].shape[0])]
    for j, tempY, temp_beta in zip(range(output_num), Y, beta):
        for i, x, y in zip(range(tempY.shape[0]), X, tempY):
            xhat = np.concatenate((np.array([x]).T, np.array([[1]])))
            p[i][j] = p1(xhat, temp_beta)
    yhat = [tp.index(max(tp)) for tp in p]
    Y = [i.T[0] for i in Y]
    Y = np.array(Y).T.tolist()
    ydesire = [ty.index(max(ty)) for ty in Y]
    for i in range(len(yhat)):
        if yhat[i] == ydesire[i]:
            correct_cnt += 1
    return correct_cnt


import pandas as pd
data = pd.read_csv('iris.data', sep=',',
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
data = data.sample(frac=1)
dataX = (data.loc[:, ['sepal_length', 'sepal_width',
                      'petal_length', 'petal_width']])
dataX = (dataX - dataX.min())/(dataX.max() - dataX.min())

input_dim = 4
output_dim = 1
output_num = 3
beta = [np.random.normal(size=(input_dim + 1, output_dim))] * 3


X = dataX.loc[:, ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width']].as_matrix()
Y = pd.get_dummies(data.loc[:, ['class']], columns=['class']).as_matrix().T
Y = [np.array([y]).T for y in Y]


def k_fold(X, Y, beta, k):
    step = (X.shape[0] + 1) // k
    correct_cnt = 0
    for i in range(k):
        print('the',i,'th fold train started')
        for j, tempY, temp_beta in zip(range(output_num), Y, beta):
            # pdb.set_trace()
            train_X = np.concatenate((X[:i*step, :], X[(i+1)*step:, :]))
            train_Y = np.concatenate(
                (tempY[:i*step, :], tempY[(i+1)*step:, :]))
            beta[j] = train(train_X, train_Y, temp_beta)
        test_X = X[i*step:min((i+1)*step, X.shape[0]), :]
        test_Y = [y[i*step:min((i+1)*step, y.shape[0]), :] for y in Y]
        correct_cnt += test(test_X, test_Y, beta)
    print(k, 'fold test finished', correct_cnt, 'was correct')
