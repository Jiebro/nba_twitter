import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import operator
from collections import Counter
from nltk.corpus import stopwords
from nltk import bigrams
import string
import vincent

# source: http://adilmoujahid.com/posts/2014/07/twitter-analytics/ text mining for twitter analytics 

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

# remove punctuation from list of most frequent words to collect
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

tweets_data_path = '../data/boston_tweets.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
# counters for each term we are interested in
count_all = Counter()
count_stop = Counter()
count_bigram = Counter()
count_hash = Counter()
count_terms_only = Counter()
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
        terms_all = [term for term in preprocess(tweet['text'])]
        terms_stop = [term for term in preprocess(tweet['text']) if term not in
        stop]
        terms_bigram = bigrams(terms_stop)
        terms_hash = [term for term in preprocess(tweet['text'])
        if term.startswith('#')]
        terms_only = [term for term in preprocess(tweet['text']) if term not in
        stop and not term.startswith(('#', '@'))]
        count_all.update(terms_all)
        count_stop.update(terms_stop)
        count_bigram.update(terms_bigram)
        count_hash.update(terms_hash)
        count_terms_only.update(terms_only)
    except:
        continue

print len(tweets_data)
print("all most common: ", count_all.most_common(5))
print("stop most common: ", count_stop.most_common(5))
print("bigram most common: ", count_bigram.most_common(5))
print("hashtag most common: ", count_hash.most_common(5))
print("no hashtag or user most common : ", count_terms_only.most_common(5))

hashtag_freq = count_all.most_common(20)
labels, freq = zip(*hashtag_freq)
data = {'data': freq, 'x': labels}
bar = vincent.Bar(data, iter_idx = 'x')
bar.to_json('hashtag_freq.json', html_out=True, html_path = 'chart.html')

tweets = pd.DataFrame()

tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)

tweets_by_lang = tweets['lang'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages tweeted (TD garden Celtics vs. Knicks)', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')
plt.savefig('nba_langs', dpi=100)

tweets_by_country = tweets['country'].value_counts()

fig2, ax2 = plt.subplots()
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=10)
ax2.set_xlabel('Countries', fontsize=15)
ax2.set_ylabel('Number of tweets' , fontsize=15)
ax2.set_title('Top 5 countries users are from (TD Garden Celtics vs. Knicks)', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax2, kind='bar', color='blue')
plt.savefig('nba_countries', dpi=100)

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

tweets['NBA'] = tweets['text'].apply(lambda tweet: word_in_text('NBA', tweet))
tweets['celtics'] = tweets['text'].apply(lambda tweet: word_in_text('celtics', tweet))
tweets['knicks'] = tweets['text'].apply(lambda tweet: word_in_text('knicks', tweet))
#
# print tweets['NBA'].value_counts()[True]
# print tweets['celtics'].value_counts()[True]
# print tweets['knicks'].value_counts()[True]
#
nba_keywords = ['NBA', 'celtics', 'knicks']
tweets_by_keywords = [tweets['NBA'].value_counts()[True], tweets['celtics'].value_counts()[True], tweets['knicks'].value_counts()[True]]

x_pos = list(range(len(nba_keywords)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_keywords, width, alpha=1, color='g')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: NBA vs. celtics vs. knicks', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(nba_keywords)
plt.grid()
plt.savefig('nba_keywords', dpi=100)
