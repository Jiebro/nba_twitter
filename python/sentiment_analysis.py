# coding=utf-8
import csv
import nltk
from nltk.probability import FreqDist

# list of positive and negative tweets for training data
pos_tweets = []
neg_tweets = []

# read data into these lists based on which example they are
with open('../data/training_data.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row.pop(0) == '0':
            neg_tweets.append((', '.join(row), 'negative'))
        else:
            pos_tweets.append((', '.join(row), 'positive'))

print neg_tweets

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
	words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((words_filtered, sentiment))
print tweets

def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features = wordlist.keys()
	return word_features

def get_words_in_tweets(tweets):
	all_words = []
	for (words, sentiment) in tweets:
		all_words.extend(words)
	return all_words

word_features = get_word_features(get_words_in_tweets(tweets))
print word_features

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

training_set = nltk.classify.apply_features(extract_features, tweets)
print training_set
classifier = nltk.NaiveBayesClassifier.train(training_set)
print classifier.show_most_informative_features(32)
