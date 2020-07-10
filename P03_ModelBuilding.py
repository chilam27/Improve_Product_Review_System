# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:37:18 2020

@author: Chi Lam
"""

#Import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


#Read in data
df = pd.read_csv('review_cleanned.csv')


#Vectorization with Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams: bi-gram and tri-gram
ngram_cv = TfidfVectorizer(ngram_range=(2,2))
X = ngram_cv.fit_transform(df.review_cleaned)
df_x = pd.DataFrame(X.toarray(), columns=ngram_cv.get_feature_names())

df = pd.concat([df, df_x], axis = 1)

#Splitting test and train data set
X = df.drop(['rating', 'customer_id', 'customer_name', 'review_header', 'review_body', 'review_txt', 'review_cleaned'], axis = 1)
y = df.rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


#Classification model
##Logistic regression
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1).fit(X_train, y_train)
print ("Accuracy: ", accuracy_score(y_train, log_reg.predict(X_train))) #0.5545

#Naive Bayes
naive_bayes = GaussianNB().fit(X_train, y_train)
print ("Accuracy: ", accuracy_score(y_train, naive_bayes.predict(X_train))) #0.9472

##Random forest classifier
randomfor_reg = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
print ("Accuracy: ", accuracy_score(y_train, randomfor_reg.predict(X_train))) #0.5393

##Support vector machines (SVMs)
supportvector = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)
print ("Accuracy: ", accuracy_score(y_train, supportvector.predict(X_train))) #0.5393

##K-Nearest neighbor
k_neighbor = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
print ("Accuracy: ", accuracy_score(y_train, k_neighbor.predict(X_train))) #0.5798


#Confusion matrix for evaluation metrics
