import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn import metrics
from pdb import set_trace

"""********************************************************
                    data instantiation
********************************************************"""

data = pd.read_csv("data/data.csv")
stopwords = stopwords.words()

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
                    K means model training
********************************************************"""
model = Word2Vec(corpus,min_count=10)
vocabulary = model.wv.vocab
X = model[vocabulary]
K = 5
kmeans_model = KMeans(n_clusters=K).fit(X)
cluster_centers = kmeans_model.cluster_centers_


"""********************************************************
                    dimension reduction 
********************************************************"""

pca = PCA(n_components=2)
result = pca.fit_transform(X)

x1 = np.array(result[:,0])
x2 = np.array(result[:,1])

# create new plot and data
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

"""********************************************************
                    2-d scatter plot
********************************************************"""

# KMeans algorithm for 2-d
kmeans_model = KMeans(n_clusters=K).fit(X)
centers = np.array(kmeans_model.cluster_centers_)
print("2d centers", centers)


colors = ["b", "g", "r", "c", "m"]
markers = ['o', 'v', 's', "*", "h"]

plt.plot()
plt.title('k means centroids')

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')

plt.scatter(result[:,0],result[:,1])
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')

words = list(vocabulary)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i,0], result[i,1]))

plt.show()
set_trace()