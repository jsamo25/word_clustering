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

model = Word2Vec(corpus,min_count=3)
vocabulary = model.wv.vocab
X = model[vocabulary]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

x1 = np.array(result[:,0])
x2 = np.array(result[:,1])

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'c']
markers = ['o', 'v', 's']

# KMeans algorithm
K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)

print(kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)

plt.plot()
plt.title('k means centroids')

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')

plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')



plt.scatter(result[:,0],result[:,1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i,0], result[i,1]))

plt.show()