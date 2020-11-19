import re
import pickle
import pandas as pd

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from pdb import set_trace

"""********************************************************
                    data instantiation
********************************************************"""

"""Tweeets data"""
#data = pd.read_csv("data/data.csv")#[:1000]

"""Movie reviews data"""
data = pd.read_csv("data/IMDB Dataset.csv")
data = data.rename(columns={"review":"text"})

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
                 K means model training
********************************************************"""

word2vec_model = Word2Vec(corpus,min_count=250)
#word2vec = word2vec_model.wv
vocabulary = word2vec_model.wv.vocab
X = word2vec_model[vocabulary]

kmeans_model = KMeans(n_clusters=7)
kmeans_model.fit(X)

cluster_centers = kmeans_model.cluster_centers_
cluster_inertia = kmeans_model.inertia_
cluster_labels = kmeans_model.labels_

"""********************************************************
                 K means model export
********************************************************"""

pickle.dump(kmeans_model, open("model/kmeans_model.pkl", 'wb'))
pickle.dump(word2vec_model,open("model/word2vec_model.pkl", 'wb'))

"""********************************************************
                 K means cluster analysis
********************************************************"""

words = [word[0] for word in list(vocabulary.items())]

labeled_data = {}
for i,word in enumerate(words):
    labeled_data[word] = cluster_labels[i]

#all words in cluster 1
for cluster in list(set(cluster_labels)):
    print()
    print("Words contained in cluster {}".format(cluster))
    print([word for word, label in labeled_data.items() if label == cluster])

"""********************************************************
                 
********************************************************"""
