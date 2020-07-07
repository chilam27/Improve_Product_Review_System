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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Read in data
df = pd.read_csv('review_scraped.csv')


#Combine 'review_header' and 'review_body'
df['review_txt'] = df.review_header + ' ' + df.review_body


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
        
    text = re.sub(r'[-();:.,?!"|0-100]','', text) # remove punctuations
    
    text = word_tokenize(text) # tokenization
    
    stop_word = stopwords.words('english')
    text = [x for x in text if x not in stop_word] # remove stop words
    
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(x[0], get_wordnet_pos(x[1])) for x in pos_tags] # lemmatization
    
    text = [x for x in text if len(x) > 1] # only get word that has more than one character
    
    return text
    
cleaned_text = df.review_txt.apply(lambda x: clean_text(x))
df['review_cleaned'] = cleaned_text


#Split data to test and training set


#Vectorization with n-gram
ngram_cv = CountVectorizer(binary=True, ngram_range=(1, 3))
ngram_cv.fit(df.review_cleaned)


#Feature engineering: sentiment analysis

                           
