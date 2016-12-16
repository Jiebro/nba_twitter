# coding=utf-8
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import csv
import json
import string
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
import random
from random import randint

# source: http://brandonrose.org/clustering document clustering for python

cavs_data_path = '../data/cavalier_starters.txt'
mavs_data_path = '../data/Mavs.txt'
raptors_data_path = '../data/raptors_starters.txt'
sixers_data_path = '../data/76ers.txt'

punctuation = list(string.punctuation)
stopwords = nltk.corpus.stopwords.words('english') + punctuation + ['rt', 'via',
'RT']

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

datapaths = [cavs_data_path, mavs_data_path, raptors_data_path, sixers_data_path]

def merge_tweets(datapaths):
    merged_data = []
    for i in range(0, len(datapaths)):
        data_file = open(datapaths[i], "r")
        for line in data_file:
            try:
                tweet = json.loads(line)
                merged_data.append(tweet['text'])
            except:
                continue
    return merged_data

test = merge_tweets(datapaths)

cavs_data= load_data(cavs_data_path)
cavs = len(cavs_data)
mavs_data = load_data(mavs_data_path)
mavs = len(mavs_data)
raptors_data = load_data(raptors_data_path)
raptors = len(raptors_data)
sixers_data = load_data(sixers_data_path)
sixers = len(sixers_data)

teams = []
for i in range(0, len(cavs_data)):
    teams.insert(i, "Cavaliers")
for i in range(cavs+1, mavs+cavs+1):
    teams.insert(i, "Mavericks")
for i in range(mavs+cavs+2, mavs+cavs+raptors+2):
    teams.insert(i, "Raptors")
for i in range(mavs+cavs+raptors+3, mavs+cavs+raptors+sixers+3):
    teams.insert(i, "Sixers")

teams2 = []
for i in range(0, len(cavs_data)):
    teams2.append(i)

# print(len(teams) == len(test))

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
    if word not in stopwords and not word.startswith(('https', '/', '.'))]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
    if word not in stopwords and not word.startswith(('https', '/', '.', '@'))]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in test:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
# # print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame)
# #
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.15, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
# # #
tfidf_matrix = tfidf_vectorizer.fit_transform(test) #fit the vectorizer to tweets
# # #
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)
# #
num_clusters = 3
#
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
# # #
joblib.dump(km,  'doc_cluster.pkl')
# # # #
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

tweets = {'cluster': clusters, 'title': teams}
frame = pd.DataFrame(tweets, index = [clusters] , columns = ['cluster', 'title'])
print(frame['cluster'].value_counts())
# # # #
print("Top terms per cluster:")
print()

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

# # #
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]
#
keys = range(0, 3)
colors = []
for i in range(3):
    colors.append("#%06x" % random.randint(0, 0xFFFFFF))

cluster_colors = dict(zip(keys, colors))
print(cluster_colors)

name_keys = range(0,3)
cluster_names = []
# for i in range(7):
#     cluster_names.append('Clust' + str(i))
cluster_names = ['LeBron James',
'Guessing game outcome', 'Dwayne Wade']

cluster_names = dict(zip(name_keys, cluster_names))
print(cluster_names)

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=teams))
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05)
#
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
#
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.savefig("clusts.png")
