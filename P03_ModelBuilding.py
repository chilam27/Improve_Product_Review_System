# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:37:18 2020

@author: Chi Lam
"""

#Import modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


#Read in data


#Vectorization with Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams: bi-gram and tri-gram
ngram_cv = TfidfVectorizer(ngram_range=(2,3))
X = ngram_cv.fit_transform(df.review_cleaned)
df_x = pd.DataFrame(X.toarray(), columns=ngram_cv.get_feature_names())

df = pd.concat([df, df_x], axis = 1)