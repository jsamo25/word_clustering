import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from pdb import set_trace

"""********************************************************
                    data instantiation
********************************************************"""

data = pd.read_csv("data/data.csv")
stopwords = stopwords.words()
stopwords_extra = ["rt", "follow", "mention","http"]
stopwords.append(stopwords_extra[0])

def clean_sentence(context):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', context).lower().split(" ")

    for word in list(sentence):
        if word in stopwords:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence

data["text"] = data["text"].apply(clean_sentence)
corpus = [sentence.split() for sentence in data["text"]]


"""********************************************************
               K means model training: 300D
********************************************************"""

model = Word2Vec(corpus,min_count=50)
vocabulary = model.wv.vocab
X = model[vocabulary]
K = 5
kmeans_model = KMeans(n_clusters=K).fit(X)
cluster_centers = kmeans_model.cluster_centers_

"""********************************************************
                 dimension reduction: 2D
********************************************************"""

pca = PCA(n_components=2)
components = pca.fit_transform(X)

x = np.array(components[:,0])
y = np.array(components[:,1])
space_2d = np.array(list(zip(x, y))).reshape(len(x), 2)
kmeans_model = KMeans(n_clusters=K).fit(space_2d)

"""********************************************************
                    scatter plot: 2D
********************************************************"""

centers = np.array(kmeans_model.cluster_centers_)
colors = ["b", "g", "r", "c", "m"]
markers = ['o', 'v', 's', "+", "D"]

plt.plot()
plt.title('k means centroids')

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x[i], y[i], color=colors[l], marker=markers[l],ls='None')

#plt.scatter(components[:,0],components[:,1])
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')

words = list(vocabulary)
for i, word in enumerate(words):
    plt.annotate(word, xy=(components[i,0], components[i,1]))

#plt.show()
#TODO: print all words but label only the 10 closest to the centroid.
"""********************************************************
                 dimension reduction: 3D
********************************************************"""

X = model[vocabulary]
pca = PCA(n_components=3)
components = pca.fit_transform(X)

x = np.array(components[:,0])
y = np.array(components[:,1])
z = np.array(components[:,2])

space_3d = np.array(list(zip(x, y, z))).reshape(len(x), 3)
kmeans_model = KMeans(n_clusters=K).fit(space_3d)

"""********************************************************
                   scatter plot: 3D
********************************************************"""

centers = np.array(kmeans_model.cluster_centers_)
ax = plt.axes(projection="3d")
colors = ["b", "g", "r", "c", "m"]
markers = ['o', 'v', 's', "+", "D"]

for i, l in enumerate(kmeans_model.labels_):
    ax.scatter3D(x[i], y[i], z[i],color=colors[l], marker=markers[l],ls='None')
#ax.scatter3D(components[:,0],components[:,1],components[:,2])
ax.scatter3D(centers[:,0], centers[:,1],centers[:,2], marker="x", color='r')

words = list(vocabulary)
for i, word in enumerate(words):
    ax.text(components[i,0],components[i,1],components[i,2],
            "%s" % (str(word)), size=8,zorder=1, color='k')
#plt.show()

"""********************************************************
                       Animated 3D
********************************************************"""

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)


