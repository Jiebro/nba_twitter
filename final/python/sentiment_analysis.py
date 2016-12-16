# coding=utf-8
import csv
import collections
from collections import Counter
import string
import json
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from vaderSentiment import sentiment as vaderSentiment

# sources: http://www.nltk.org/book/ch06.html naive bayes classifier,
# https://medium.com/@aneesha/quick-social-media-sentiment-analysis-with-vader-da44951e4116#.hzsth01he
# vader sentiment analysis 

# list of positive and negative tweets for training data
pos_tweets = []
neg_tweets = []

# read data into these lists based on positive, negative examples
with open('../data/training_data.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row.pop(0) == '0':
            neg_tweets.append((', '.join(row), 'negative'))
        else:
            pos_tweets.append((', '.join(row), 'positive'))

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

# append words in tweets to list with sentiment
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
	words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((words_filtered, sentiment))
# print tweets

# classifier features
def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features = wordlist.keys()
	return word_features

# extract words from tweets and sentiment for training set
def get_words_in_tweets(tweets):
	all_words = []
	for (words, sentiment) in tweets:
		all_words.extend(words)
	return all_words

word_features = get_word_features(get_words_in_tweets(tweets))
# print word_features

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

training_set = nltk.classify.apply_features(extract_features, tweets)
# print training_set
nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
# print nb_classifier.show_most_informative_features(50)

# read tweets from cavs to be classified into an array with location
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

# read csv of tweets collected from team homepages
def read_csv(datapath):
    tweets_data = []
    with open(datapath, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            tweets_data.append(row[2])
    return tweets_data

# populate lists of tweets classified as positive, negative and extract location
def classify(tweets, classifier):
    pos_data = []
    neg_data = []
    for tweet in tweets:
        classLabel = classifier.classify(extract_features(tweet['text'].split()))
        if(classLabel == 'positive'):
            pos_data.append(tweet['place'])
        else:
            neg_data.append(tweet['place'])
    return [pos_data, neg_data]

def classify_team(tweets, classifier):
    pos_data = []
    neg_data = []
    for tweet in tweets:
        classLabel = classifier.classify(extract_features(tweet.split()))
        if(classLabel == 'positive'):
            pos_data.append(tweet)
        else:
            neg_data.append(tweet)
    return [pos_data, neg_data]

# get team data for classifying
cavs_data_path = '../data/cavalier_starters.txt'
cavs_data = load_data(cavs_data_path)


for sentence in cavs_data:
    print sentence,
    sentiment = vaderSentiment(sentence)
    print "\n\t" + str(sentiment)
# cavs_pos = classify(cavs_data, nb_classifier)[0]
# cavs_neg = classify(cavs_data, nb_classifier)[1]
#
# raptors_data_path = '../data/raptors_starters.txt'
# raptors_data = load_data(raptors_data_path)
# raptors_pos = classify(raptors_data, nb_classifier)[0]
# raptors_neg = classify(raptors_data, nb_classifier)[1]
#
# seventy_sixers_data_path = '../data/76ers.txt'
# seventy_sixers_data = load_data(seventy_sixers_data_path)
# seventy_sixers_pos = classify(seventy_sixers_data, nb_classifier)[0]
# seventy_sixers_neg = classify(seventy_sixers_data, nb_classifier)[1]
#
# mavs_data_path = '../data/Mavs.txt'
# mavs_data = load_data(mavs_data_path)
# mavs_pos = classify(mavs_data, nb_classifier)[0]
# mavs_neg = classify(mavs_data, nb_classifier)[1]
def tokenize(s):
    return tokens_re.findall(s)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'RT']

ATL_data_path = "../data/ATLHawks_tweets.csv"
ATL_data = read_csv(ATL_data_path)
ATL_pos = classify_team(ATL_data, nb_classifier)[0]
# ATL_neg = classify_team(ATL_data, nb_classifier)[1]
# print len(ATL_data)
# tweets_file = open(ATL_data_path, "r")
# count_all = Counter()
# for tweet in tweets_file:
#         terms_all = [term for term in preprocess(tweet) if term not in stop]
#         count_all.update(terms_all)
#
# print(count_all.most_common(5))

# positive to negative ratio for data set
def get_ratio_pos(pos_ex, neg_ex):
    total = float(len(pos_ex) + len(neg_ex))
    pos = float(len(pos_ex))
    neg = float(len(neg_ex))
    pos_to_neg = float(pos/neg)
    return pos_to_neg

# print get_ratio_pos(raptors_pos, raptors_neg)
# print get_ratio_pos(cavs_pos, cavs_neg)
# print get_ratio_pos(seventy_sixers_pos, seventy_sixers_neg)
# print get_ratio_pos(mavs_pos, mavs_neg)
