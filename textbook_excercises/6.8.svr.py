from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd

data_frame = pd.read_csv('西瓜3.0alpha.txt', sep=' ', encoding='gb2312')
feature = data_frame[['密度']].as_matrix()
label = data_frame['含糖率'].as_matrix()

linear_svr = svm.SVR(C=10000, kernel='linear')
linear_svr.fit(feature, label)

linear_accuracy = 0
for x, y in zip(feature, label):
    yhat = linear_svr.predict(x.reshape(1, -1))
    print(y, yhat)
    if yhat == y:
        linear_accuracy += 1

gaussian_svr = svm.SVR(C=10000, kernel='rbf')
gaussian_svr.fit(feature, label)
gaussian_accuracy = 0
for x, y in zip(feature, label):
    yhat = gaussian_svr.predict(x.reshape(1, -1))
    print(y, yhat)
    if yhat == y:
        gaussian_accuracy += 1

print('linear_accuracy:', linear_accuracy / len(data_frame))
print('gaussian_accuracy:', gaussian_accuracy / len(data_frame))

plt.subplot(2, 2, 1)
for x, y in zip(feature, label):
    plt.plot(x[0], y, 'r.' if y == 0 else 'b.')

plt.subplot(2, 2, 2)
plt.title('linear kernel svr support vectors')
for x, y in zip(feature, label):
    if x in linear_svr.support_vectors_:
        plt.plot(x[0], y, 'yo')
    else:
        plt.plot(x[0], y, 'r.' if y == 0 else 'b.')

plt.subplot(2, 2, 3)
plt.title('gaussian kernel svr support vectors')
for x, y in zip(feature, label):
    if x in gaussian_svr.support_vectors_:
        plt.plot(x[0], y, 'yo')
    else:
        plt.plot(x[0], y, 'r.' if y == 0 else 'b.')
