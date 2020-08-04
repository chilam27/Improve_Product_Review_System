# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:12:56 2020

@author: Chi Lam
"""

#Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud
from nltk import FreqDist
import gensim
from gensim import corpora


#Read in data
df = pd.read_csv('review_cleanned.csv')
df.info()


#Analyze 'rating'
##Univariate analysis
plt.figure(figsize=(10,5))
sns.countplot(df.rating)
plt.title('Figure 2: overall ratings counts', fontsize=15)


##Multivariate analysis
###'rating' vs. 'review_date'
plt.figure(figsize=(13,7))
sns.countplot(x="review_date", hue="rating", data=df)
plt.title('Figure 3: ratings over time counts', fontsize=15)

###'rating' vs. 'characrer_len'
plt.figure(figsize=(13,10))
plt.subplot(121)
sns.boxplot(df.rating, df.character_len)
plt.title('Figure 4: ratings vs. character length', fontsize=15)

###'rating' vs. 'word_count'
plt.subplot(122)
sns.boxplot(df.rating, df.word_count)
plt.title('Figure 5: ratings vs. word count', fontsize=15)

###'rating' vs. 'predict_sentiment'
count1 = []
count2 = []
ind = np.arange(5)
width = 0.35

for i in range(5):
    a = df[df.rating == (i+1)].shape[0]
    b = df[df.predict_sentiment == (i+1)].shape[0]
    count1.append(a)
    count2.append(b)
    
plt.figure(figsize=(8,8))
plt.bar(ind, count1, width, label = 'actual rating')
plt.bar(ind + width, count2, width, label = 'predict rating')
plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5'))
plt.legend(loc='best')
plt.title('Figure 6: rating vs. predicted sentiment')
plt.ylabel('count')
plt.xlabel('rating')
    

#Analyze words with world cloud
text = " ".join(review for review in df.review_cleaned)

plt.figure(figsize=(13,7))
wordcloud = WordCloud(max_font_size=100, max_words=200, background_color="white", random_state = 1).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Analyze rating by words frequency
lis = ['a','b','c','d','e']
for i in range (5):
    df_test = df[df.rating == (i+1)]
    df_test.review_cleaned_token = df_test.review_cleaned.apply(lambda x: x.split(' '))
    star = df_test.review_cleaned_token.apply(pd.Series).stack()
    freq_dist_pos = FreqDist(star)

    plt.figure(figsize=(10, 6))
    title = 'Figure 8' + lis[i] + ': top 20 word frequency for ' + str(i+1) + ' stars reviews'
    freq_dist_pos.plot(20, cumulative=False, title = title)


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
