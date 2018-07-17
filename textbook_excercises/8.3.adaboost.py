import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data_frame = pd.read_csv('西瓜3.0alpha.txt', sep=' ', encoding='gb2312')
column_name = data_frame.columns
feature = data_frame[['密度', '含糖率']].as_matrix().astype('float32')
label = data_frame['好瓜'].as_matrix()

adaboost = AdaBoostClassifier(DecisionTreeClassifier())
adaboost.fit(feature, label)


plot_colors = "br"
plot_step = 0.01
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = feature[:, 0].min(), feature[:, 0].max()
y_min, y_max = feature[:, 1].min(), feature[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# for x, y, z in zip(xx.ravel(), yy.ravel(), Z.ravel()):
#     plt.plot(x, y, 'r.' if z == 0 else 'b.')

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(label == i)
    plt.scatter(feature[idx, 0], feature[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

