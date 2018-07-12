import numpy as np
from math import log
from math import exp


input_dim = 2
output_dim = 1

beta = np.random.normal(size=(input_dim + 1, output_dim))


def p1(xhat, beta):
    e = np.exp(np.dot(beta.T, xhat))[0][0]
    return e / (1 + e)


def iterate(X, Y, beta):
    import pdb
    # pdb.set_trace()
    grad = np.zeros(shape=beta.shape)
    grad2 = 0
    loss = 0
    for x, y in zip(X, Y):
        xhat = np.concatenate((np.array([x]).T, np.array([[1]])))
        grad += - xhat * (y - p1(xhat, beta))
        grad2 += np.dot(xhat, xhat.T) * p1(xhat, beta) * (1 - p1(xhat, beta))
        loss += log(1 + exp(np.dot(beta.T, xhat))) - y * np.dot(beta.T, xhat)
        print(log(1 + exp(np.dot(beta.T, xhat))) - y * np.dot(beta.T, xhat))
        # pdb.set_trace()
    beta = beta - np.dot(np.linalg.inv(grad2), grad)
    return grad, grad2, beta, loss


X = np.array([[.697, .460], [.774, .376],
              [.634, .264], [.608, .318],
              [.556, .215], [.403, .237],
              [.481, .149], [.437, .211],
              [.666, .091], [.243, .267],
              [.245, .057], [.343, .099],
              [.639, .161], [.657, .198],
              [.360, .370], [.593, .042], [.719, .103]])
Y = np.array([[1]] * 8 + [[0]] * 9)

epoch = 50
for i in range(epoch):
    print('Epoch' ,i ,'started')
    grad, grad2, beta, loss = iterate(X, Y, beta)
    print('loss =',loss)
