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
            neg_tweets.append(row)
        else:
            pos_tweets.append(row)

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
	words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((words_filtered, sentiment))

print tweets
