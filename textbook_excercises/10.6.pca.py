import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import decomposition
import numpy as np
from mxnet import gluon
import os

picture_files = os.listdir('yalefaces/')
pictures = []
for picture_file in picture_files:
    if picture_file != 'Readme.txt':
        pictures.append(mpimg.imread('yalefaces/' + picture_file))
data = np.concatenate([picture.reshape((1, -1)) for picture in pictures], axis=0)

pca = decomposition.PCA(20)
pca.fit(data)
eigen_faces = pca.components_.reshape((20, 243, 320))

for eigen_face in eigen_faces:
    plt.imshow(eigen_face)
    plt.show()
