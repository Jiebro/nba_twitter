# coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import csv
import json
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

data_path = '../data/76ers.txt'

def load_data(datapath):
    tweets_data = []
    data_file = open(datapath, "r")
    for line in data_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet['text'])
        except:
            continue
    return tweets_data

sixers_data = load_data(data_path)
print sixers_data

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sixers_data)
#
true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print
