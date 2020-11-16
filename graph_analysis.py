import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

"""********************************************************
               import trained model
********************************************************"""

word2vec_model = pickle.load(open("model/word2vec_model.pkl", 'rb'))
kmeans_model = pickle.load(open("model/kmeans_model.pkl", "rb"))

vocabulary = word2vec_model.wv.vocab
X = word2vec_model[vocabulary]

words = list(vocabulary)
colors = ["b", "g", "r", "c", "m"]
markers = ['o', 'v', 's', "+", "D"]

"""********************************************************
                 dimension reduction: 2D
********************************************************"""

pca = PCA(n_components=2)
components = pca.fit_transform(X)

x = np.array(components[:,0])
y = np.array(components[:,1])
space_2d = np.array(list(zip(x, y))).reshape(len(x), 2)
kmeans_model = KMeans(n_clusters=5).fit(space_2d)
centers = np.array(kmeans_model.cluster_centers_)

"""********************************************************
                    scatter plot: 2D
********************************************************"""

plt.plot()
plt.title('k means centroids')
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x[i], y[i], color=colors[l], marker=markers[l],ls='None')

for i, word in enumerate(words):
    plt.annotate(word, xy=(components[i,0], components[i,1]))

#plt.show()

"""********************************************************
                 dimension reduction: 3D
********************************************************"""

pca = PCA(n_components=3)
components = pca.fit_transform(X)

x = np.array(components[:,0])
y = np.array(components[:,1])
z = np.array(components[:,2])

space_3d = np.array(list(zip(x, y, z))).reshape(len(x), 3)
kmeans_model = KMeans(n_clusters=5).fit(space_3d)
centers = np.array(kmeans_model.cluster_centers_)

"""********************************************************
                   scatter plot: 3D
********************************************************"""

ax = plt.axes(projection="3d")
ax.scatter3D(centers[:,0], centers[:,1],centers[:,2], marker="x", color='r')

for i, l in enumerate(kmeans_model.labels_):
    ax.scatter3D(x[i], y[i], z[i],color=colors[l], marker=markers[l],ls='None')

for i, word in enumerate(words):
    ax.text(components[i,0],components[i,1],components[i,2],
            "%s" % (str(word)), size=8,zorder=1, color='k')
#plt.show()

"""********************************************************
                       360 rotation
********************************************************"""

# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
