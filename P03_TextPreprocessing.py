# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:52:01 2020

@author: Chi Lam
"""

#Import modules
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Read in data
df = pd.read_csv('review_scraped.csv')

df.info()
df.describe()

#Combine and delete columns
df['review_txt'] = df.review_header + ' ' + df.review_body

df['verified_purchase'].value_counts()
df['review_loc'].value_counts()
del df['verified_purchase'] # because all reviews we got are verified purchase so there is no point analyzing
del df['review_loc'] # because all reviews we got are from the United States so there is no point analyzing


#Generalize 'date'
df.review_date.value_counts()
df.review_date = df.review_date.apply(lambda x: x.split(' ')[-1]) # group the month together to form quarter period of a year
    

#Cleaning text/ text preprocessing and text normalization
def get_wordnet_pos(tag):
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def decontraction(text_string):
    text_string = re.sub(r"n\'t", " not", text_string)
    text_string = re.sub(r"\'re", " are", text_string)
    text_string = re.sub(r"\'s", " is", text_string)
    text_string = re.sub(r"\'d", " would", text_string)
    text_string = re.sub(r"\'ll", " will", text_string)
    text_string = re.sub(r"\'t", " not", text_string)
    text_string = re.sub(r"\'ve", " have", text_string)
    text_string = re.sub(r"\'m", " am", text_string)
    return text_string

def clean_text(text):
    text = text.lower() # convert text to lowercase
    
    text = decontraction(text) # replace contractions with their longer forms
    
    text = re.sub(r'[-();:.,?!"[0-9]+','', text) # remove punctuations and numbers
    
    text = word_tokenize(text) # tokenization
    
    stop_word = stopwords.words('english')
    text = [x for x in text if x not in stop_word] # remove stop words
    
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(x[0], get_wordnet_pos(x[1])) for x in pos_tags] # lemmatization
    
    text = [x for x in text if len(x) > 1] # only get word that has more than one character
    
    return text

df['review_cleaned'] = df.review_txt.apply(lambda x: " ".join(clean_text(x)))

for i in range(len(df.review_cleaned)):
    if len(df.review_cleaned[i]) == 0:
        df.drop(index=i, inplace=True) # dropping 'review_cleaned' column with empty value


#Feature engineering: sentiment anaylsis usingVadar
sentiment = SentimentIntensityAnalyzer()

def sentiment_analysis(compound):
    if compound >= 0.6:
        return '5'
    elif 0.6 > compound >= 0.2:
        return '4'
    elif 0.2 > compound >= -0.2:
        return '3'
    elif -0.2 > compound >= -0.4:
        return '2'
    else:
        return '1'

df['predict_sentiment'] = df.review_cleaned.apply(lambda x: sentiment_analysis(sentiment.polarity_scores(x)['compound']))


#Add length of character and word columns:
df['character_len'] = df.review_cleaned.apply(lambda x: len(x))

df['word_count'] = df.review_cleaned.apply(lambda x: len(x.split(' ')))


#Export to csv file
df.to_csv('review_cleanned.csv', index=False)