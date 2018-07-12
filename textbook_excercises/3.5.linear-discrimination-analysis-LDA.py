import numpy as np
from math import log
from math import exp
from matplotlib import pyplot as plt


input_dim = 2
output_dim = 1

# define the dataset
rawX = np.array([[.697, .460], [.774, .376],
              [.634, .264], [.608, .318],
              [.556, .215], [.403, .237],
              [.481, .149], [.437, .211],
              [.666, .091], [.243, .267],
              [.245, .057], [.343, .099],
              [.639, .161], [.657, .198],
              [.360, .370], [.594, .042], [.719, .103]])
rawY = np.array([[1]] * 8 + [[0]] * 9, dtype='int16')

# convert to the LDA dataset
X = [np.zeros(shape=(0,2)), np.zeros(shape=(0,2))]
for i,x in enumerate( rawX ):
    y = rawY[i]
    X[y[0]] = np.concatenate((X[y[0]], np.array([ x ])), axis=0)

# calculate the average of vectors of each class
miu = [np.array( [ x.mean(axis=0) ] ) for x in X]

# calculate the within-class scatter matrix
sw = np.zeros(shape=(X[0].shape[1],)*2)
for x,miui in zip(X,miu):
    for j in range( len(x) ):
        xj = x[j:j+1, :] - miui
        sw += np.dot(xj.T, xj)

# calculate the desired weight
w = np.dot( np.linalg.inv(sw), (miu[0] - miu[1]).T )

k=w[1]/w[0]
k=k[0]

for x in X[0]:
    plt.plot(x[0],x[1],'r.')
for x in X[1]:
    plt.plot(x[0],x[1],'b.')
plt.plot([0,.1],[0,.1*k])
plt.show()
