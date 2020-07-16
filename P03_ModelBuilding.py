# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:37:18 2020

@author: Chi Lam
"""

#Import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import matplotlib.pyplot as plt
%matplotlib inline


#Read in data
df = pd.read_csv('review_cleanned.csv')
df.describe()


#Vectorization
##Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams: bi-gram
ngram_cv = TfidfVectorizer(ngram_range=(2,2))
X = ngram_cv.fit_transform(df.review_cleaned)
df_x = pd.DataFrame(X.toarray(), columns=ngram_cv.get_feature_names())

df = pd.concat([df, df_x], axis = 1)


#Splitting test and train data set
X = df.drop(['rating', 'customer_id', 'customer_name', 'review_header', 'review_body', 'review_txt', 'review_cleaned'], axis = 1)
y = df.rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=df.rating, random_state=1)


#Normalization
norm = MinMaxScaler().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
                
#Classification model
##Logistic regression
for x in [0.01, 0.05, 0.25, 0.5, 1]:
    log_reg = LogisticRegression(C=x, solver='lbfgs', multi_class='multinomial', random_state=1).fit(X_train, y_train)
    print ("Logistics regression accuracy (tfidf) for ", x, ":", accuracy_score(y_train, log_reg.predict(X_train)))

log_reg = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', random_state=1).fit(X_train, y_train)
print ("Logistics regression accuracy (tfidf) for C=1:", accuracy_score(y_train, log_reg.predict(X_train))) #0.5533 #Normalized: 0.9872

#Naive Bayes
naive_bayes = GaussianNB().fit(X_train, y_train)
print ("Naive Bayes accuracy: ", accuracy_score(y_train, naive_bayes.predict(X_train))) #0.9452 #Normalized: 0.9204

##Random forest classifier
randomfor_reg = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
print ("Random forest classifier accuracy: ", accuracy_score(y_train, randomfor_reg.predict(X_train))) #0.5368 #Normalized: 0.5368

##K-Nearest neighbor
k_neighbor = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
print ("K-Nearest neighbor accuracy: ", accuracy_score(y_train, k_neighbor.predict(X_train))) #0.5838 #Normalized: 0.5535

##Support vector machines (SVMs)
supportvector = svm.SVC(decision_function_shape="ovo", random_state=1).fit(X_train, y_train)
print ("SVM accuracy: ", accuracy_score(y_train, supportvector.predict(X_train))) #0.5393 #Normalized: 0.9109

#Predict Test set
log_reg_test = log_reg.predict(X_test)
naive_bayes_test = naive_bayes.predict(X_test)
randomfor_reg_test = randomfor_reg.predict(X_test)
k_neighbor_test = k_neighbor.predict(X_test)
supportvector_test = supportvector.predict(X_test)

print('Log regression: ', accuracy_score(y_test, log_reg_test)) #Best one 0.6396
print('Naive Bayes: ', accuracy_score(y_test, naive_bayes_test))
print('Random forest regression: ', accuracy_score(y_test, randomfor_reg_test))
print('K-nearest neighbor: ', accuracy_score(y_test, k_neighbor_test))
print('Support vector machines: ', accuracy_score(y_test, supportvector_test))

for x in range(100):
    a = randomfor_reg.predict(np.array(list(X_test.iloc[x,:])).reshape(1,-1))[0]
    b = y_test.tolist()
    print(b[x], a)

#Confusion matrix for evaluation metrics
confusion_matrix(y_test, log_reg_test)

print(classification_report(y_test,log_reg_test))

#AUC (Area Under The Curve) - ROC (Receiver Operating Characteristics) curve. Code by Serafeim Loukas
y_bin = label_binarize(y, classes=[1, 2, 3, 4, 5])
y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
n_classes = y_bin.shape[1]

y_score = OneVsRestClassifier(log_reg).fit(X_train, y_train).decision_function(X_test)

##Plotting and estimation of FPR, TPR
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
colors = cycle(['blue', 'red', 'green', 'orange', 'yellow'])

plt.figure(figsize=(13,7))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure n: receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
