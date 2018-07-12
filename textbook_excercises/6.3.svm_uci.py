from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd

data_frame = pd.read_csv('iris.data', sep=',', encoding='gb2312',
                         names=['a', 'b', 'c', 'd', 'label'])
feature = data_frame[['a', 'b', 'c', 'd']].as_matrix()
label = data_frame['label'].as_matrix()

linear_svm = svm.SVC(C=10000, kernel='linear')
linear_svm.fit(feature, label)

linear_accuracy = 0
for x, y in zip(feature, label):
    yhat = linear_svm.predict(x.reshape(1, -1))
    print(y, yhat)
    if yhat == y:
        linear_accuracy += 1

gaussian_svm = svm.SVC(C=10000, kernel='rbf')
gaussian_svm.fit(feature, label)
gaussian_accuracy = 0
for x, y in zip(feature, label):
    yhat = gaussian_svm.predict(x.reshape(1, -1))
    print(y, yhat)
    if yhat == y:
        gaussian_accuracy += 1

print('linear_accuracy:', linear_accuracy / len(data_frame))
print('gaussian_accuracy:', gaussian_accuracy / len(data_frame))

plt.subplot(2, 2, 1)
for x, y in zip(feature, label):
    plt.plot(x[0], x[1], 'r.' if y == 0 else 'b.')

plt.subplot(2, 2, 2)
plt.title('linear kernel svm support vectors')
for x, y in zip(feature, label):
    if x in linear_svm.support_vectors_:
        plt.plot(x[0], x[1], 'yo')
    else:
        plt.plot(x[0], x[1], 'r.' if y == 0 else 'b.')

plt.subplot(2, 2, 3)
plt.title('gaussian kernel svm support vectors')
for x, y in zip(feature, label):
    if x in gaussian_svm.support_vectors_:
        plt.plot(x[0], x[1], 'yo')
    else:
        plt.plot(x[0], x[1], 'r.' if y == 0 else 'b.')
