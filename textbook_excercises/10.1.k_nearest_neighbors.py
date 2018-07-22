from sklearn import neighbors
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

data_frame = pd.read_csv('西瓜3.0alpha.txt', sep=' ', encoding='gb2312')
column_name = data_frame.columns
feature = data_frame[['密度', '含糖率']].as_matrix()
label = data_frame['好瓜'].as_matrix()

knn = neighbors.KNeighborsClassifier()
knn.fit(feature, label)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x_min, x_max = feature[:, 0].min() - 1, feature[:, 0].max() + 1
y_min, y_max = feature[:, 1].min() - 1, feature[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(feature[:, 0], feature[:, 1], c=label, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.show()
