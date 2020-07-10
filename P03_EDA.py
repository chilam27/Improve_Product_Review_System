# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:12:56 2020

@author: Chi Lam
"""

#Import modules
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud
from nltk import FreqDist
import gensim
from gensim import corpora


#Read in data
df = pd.read_csv('review_cleanned.csv')


#Analyze 'rating'
##Univariate analysis
plt.figure(figsize=(10,5))
sns.countplot(df.rating)
plt.title('Figure n: overall ratings counts', fontsize=15)


##Multivariate analysis
plt.figure(figsize=(13,7))
sns.countplot(x="review_date", hue="rating", data=df)
plt.title('Figure n: ratings over time counts', fontsize=15)


plt.figure(figsize=(13,10))
plt.subplot(121)
sns.boxplot(df.rating, df.character_len)
plt.title('Figure n: ratings vs. character length boxplots', fontsize=15)

plt.subplot(122)
sns.boxplot(df.rating, df.word_count)
plt.title('Figure n: ratings vs. word count boxplots', fontsize=15)


#Analyze words with world cloud
text = " ".join(review for review in df.review_cleaned)

plt.figure(figsize=(13,7))
wordcloud = WordCloud(max_font_size=100, max_words=200, background_color="white", random_state = 1).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Analyze rating by words frequency
def plot_word_freq(num):
    df_test = df[df.rating == num]
    df_test.review_cleaned_token = df_test.review_cleaned.apply(lambda x: x.split(' '))
    star = df_test.review_cleaned_token.apply(pd.Series).stack()
    freq_dist_pos = FreqDist(star)

    plt.figure(figsize=(10, 6))
    title = 'Figure n: top 20 word frequency for ' + str(num) + ' stars reviews'
    freq_dist_pos.plot(20, cumulative=False, title=title)

plot_word_freq(5) # five stars reviews
plot_word_freq(4) # four stars reviews
plot_word_freq(3) # three stars reviews
plot_word_freq(2) # two stars reviews
plot_word_freq(1) # one star reviews


#Latent Dirichlet Allocation (LDA) topic modeling
x = 1
while x < 6:
    df_test = df[df.rating == x]
    df_test.review_cleaned_token = df_test.review_cleaned.apply(lambda x: x.split(' '))
    star = df_test.review_cleaned_token.apply(pd.Series).stack()
    
    dictionary = corpora.Dictionary([star])

    corpus = [dictionary.doc2bow(t) for t in [star]] # initialize a corpus
    [i for i in corpus]

    model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, passes=50, random_state=1)
    print(x, 'stars ********************')
    print(model.print_topics(num_words=5))
    print(df_test.review_cleaned.head())
    print('\n')
    
    x += 1
