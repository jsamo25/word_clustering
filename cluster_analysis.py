import re
import pickle
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

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

word2vec_model = Word2Vec(corpus,min_count=100)
vocabulary = word2vec_model.wv.vocab
X = word2vec_model[vocabulary]

kmeans_model = KMeans(n_clusters=5)
kmeans_model.fit(X)
cluster_centers = kmeans_model.cluster_centers_

"""********************************************************
                         export model
********************************************************"""

pickle.dump(kmeans_model, open("model/kmeans_model.pkl", 'wb'))
pickle.dump(word2vec_model,open("model/word2vec_model.pkl", 'wb'))