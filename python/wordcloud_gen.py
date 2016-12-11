from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

d = path.dirname('../data/mavs_word.txt')

text = open(path.join(d, '../data/mavs_word.txt')).read()
basketball_mask = np.array(Image.open(path.join(d, "../figs/mavs.png")))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=basketball_mask,
stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "mavs_word_wc.jpg"))

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(basketball_mask, cmap=plt.cm.gray)
plt.axis("off")
