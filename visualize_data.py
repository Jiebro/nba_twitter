import json
import pandas as pd
import matplotlib.pyplot as plt
import re2

tweets_data_path = 'nba_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

print len(tweets_data)
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
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')
# plt.savefig('langs', dpi=100)

tweets_by_country = tweets['country'].value_counts()

fig2, ax2 = plt.subplots()
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=10)
ax2.set_xlabel('Countries', fontsize=15)
ax2.set_ylabel('Number of tweets' , fontsize=15)
ax2.set_title('Top 5 countries', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax2, kind='bar', color='blue')
# plt.savefig('countries', dpi=100)

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

tweets['NBA'] = tweets['text'].apply(lambda tweet: word_in_text('NBA', tweet))
tweets['jazz'] = tweets['text'].apply(lambda tweet: word_in_text('jazz', tweet))
tweets['knicks'] = tweets['text'].apply(lambda tweet: word_in_text('knicks', tweet))

print tweets['NBA'].value_counts()[True]
print tweets['jazz'].value_counts()[True]
print tweets['knicks'].value_counts()[True]
