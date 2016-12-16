from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

# source: https://amueller.github.io/word_cloud/auto_examples/masked.html python masked wordcloud

d = path.dirname('../data/76ers_final_text')

text = open(path.join(d, '../data/76ers_final_text')).read()
basketball_mask = np.array(Image.open(path.join(d, "../figs/basketball.jpg")))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "76ers_final_termCloud.jpg"))

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(basketball_mask, cmap=plt.cm.gray)
plt.axis("off")
