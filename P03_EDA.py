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
##Five stars reviews
df_five = df[df.rating == 5]
df_five.review_cleaned = df_five.review_cleaned.apply(lambda x: x.split(' '))
five_star = df_five.review_cleaned.apply(pd.Series).stack()
freq_dist_pos = FreqDist(five_rating)
print(freq_dist_pos.most_common(20))

##Four stars reviews
df_four = df[df.rating == 4]
df_four.review_cleaned = df_four.review_cleaned.apply(lambda x: x.split(' '))
four_star = df_four.review_cleaned.apply(pd.Series).stack()
freq_dist_pos = FreqDist(four_star)
print(freq_dist_pos.most_common(20))

##Three stars reviews
df_three = df[df.rating == 3]
df_three.review_cleaned = df_three.review_cleaned.apply(lambda x: x.split(' '))
three_star = df_three.review_cleaned.apply(pd.Series).stack()
freq_dist_pos = FreqDist(three_star)
print(freq_dist_pos.most_common(20))

##Two stars reviews
df_two = df[df.rating == 2]
df_two.review_cleaned = df_two.review_cleaned.apply(lambda x: x.split(' '))
two_star = df_two.review_cleaned.apply(pd.Series).stack()
freq_dist_pos = FreqDist(two_star)
print(freq_dist_pos.most_common(20))

##One star reviews
df_one = df[df.rating == 1]
df_one.review_cleaned = df_one.review_cleaned.apply(lambda x: x.split(' '))
one_star = df_one.review_cleaned.apply(pd.Series).stack()
freq_dist_pos = FreqDist(one_star)
print(freq_dist_pos.most_common(20))
