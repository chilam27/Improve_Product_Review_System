# Improved Product Review System

Improve the traditional product review system through sentimental analysis and Natural Language Processing (NLP). By taking only the customer review comment about the product and we can create a sentiment detection algorithm to automatically rated the product. I will also include topic modeling in this project to detect the generall topics of what the reviews are about. If the accuracy is acceptable, it will eliminate steps require customer to write a review and help them to get a quicker glance at what others are saying about the product before buying. 

## Backgorund and Motivation

I has an idea for this project from the situation I had when doing thing that most college students do when they are bored: online shopping. I am a very frugal buyer myself. That is I spend a lot of time looking at product description and its reviews to make sure I will be happy with my purchases. Besides giant e-commerce companies (such as Amazon, Etsy, Ebay, etc.), not a lot of other online retailers have anything better than a basic star and comment review system for their products. That is why, in my belief, those top tier companies are having better customer interactions. Better customer interactions can lead to seller improve their products and buyers, like me, have more confidence in buyding the product.

Combine with a new principle I just learned that is called *Principle of Least Effort* (it was first articulated by the Italian philosopher Guillaume Ferrero) and the idea that businesses improve by reducing friction that is needed for customer to buy their product (I learned it from a book: Atomic Habits by James Clear). I want to build a review system that:
  * Reduce the step of rating the product and have the machine to analyze the review comment and predict the rating instead.
  * Gather all reviews, sort out and summarize to see what topics are being mentioned.
  
Since this project touch on many of the fields I am unfunfamiliar with, there are a couple of things that I wanted to learn out of this project that I hope will benefit my journey of becoming a better data scientist:
  1. Scrape raw reviews from Amazon.
  2. Learn about text and sentimental analysis.
  4. Understand how feature engineer can give me a better outcome.
  3. Classification algorithms and the importance of feature scalling.

## Prerequisites

Python Version: 3.7.4

Packages: BeautifulSoup, requests, nltk, re, matplotlib.pyplot, seaborn, WordCloud, gensim, pandas, numpy, sklearn, itertools

## Project Outline

1. Data collection: I build a web scrapper through `BeautifulSoup` and scrape an amazon product's review. For this project, I choose a clothing item: **Dickies Men's Original 874 Work Pant**. The reason for my decision is I have more experience with the product and the quantity for the review is ideal.

2. Text preprocessing: or data cleaning; I mostly cleanned the review texts to remove noise and make it easy for the machine to read in the data.

3. Exploratory Data Analysis: I analyze the target variable ("rating") and examine its and other features' relationships. In this phase, I also perform Latent Dirichlet Allocation (LDA) topic modeling to search for topics of each rating cateogries.

4. Model Building: I compare different classification algorithms (logistic regression, Naive Bayes, random forest classifier, k-nearest neighbor (KNN), and support vector machines (SVM)) and choose the one that produce the best result. The estimators I use for my multilabel classification algorithm is the accuracy classification score that computes subset accuracy.    

### [Data Collection](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_DataCollection.py)

_*Disclaimer: this data set is used for educational puspose._

I create a web scrapper for the product reviews. My first intention was to scrape 8 different variables (see table below) plus the product's size and color. I want to see if it has any effect on determing the rating or not (such as whether such size or color has some defects). But I run into similar issue I have with my other web scrapper: that is I cannot make the function see null value as n/a and it will just skip over. So I decided to leave those two variables out of the data set. Although I want to scrape all 9,208 reviews it has, Amazon only allows to me to access to only 5,000 of the reviews. Hence, my data scrapped CSV consists of 8 variables and 5,000 records.

Variables             |  Description
:--------------------:|:---------------------------------------------------------------:
customer_id           | the unique ID of each customer
customer_name         | name of the customer
rating                | customer's rating of the product (1-5)
review_date           | review's posted date
review_loc            | customer's location (I only take review from the United States)
verified_purchase     | customer's product purchased verification
review_head           | review's title
review_body           | the main part of the review


### [Text Preprocessing](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_TextPreprocessing.py)

* Quick glance at the data set: according to figure 1, only one of our data is numerical ("rating") and the rest is categorical. Another important element is to determine whether there is any null value. Luckily for me, there is none.

<p align="center">
  <img width="500" height="300" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig1.png">
</p>

* I combine both the review's header and body together for easier analysis later on. And since both "verified_purchase" and "review_loc" only has one unique value for each variables, I remove them for the data set because it will not give us any information.

* Generalize "review_date": In order to see relationship between "rating" and "review_date", I need to group the individual date together for a more compact and reasonable graphs later on. I want to group them by quater at first, but I find that I need to compact it even more so I end up doing it by years instead.

* Then, I creat a function (_"clean_text"_) to perform text preprocessing to the review texts. I implement the following steps: convert text to lowercase, replace contractions with their longer forms, remove punctuations and numbers, tokenization (process of splitting strings into tokens), remove stop words, lemmatization(return a word to its common base root while takes into consideration the morphological analysis of the words), only get word that has more than one character. Since there will be some short reviews end up with empty value after the text is cleanned, I remove them from the data set.

* I will do a bit of feature engineering and use VADER Sentiment (_SentimentIntensityAnalyzer_) as a sentiment analysis tool to analyze the emotion of the reivew text. This tool is very good at not only determine whether a string of text is postive or negative, it also give the string a sentiment intensity score. Since the score is ranged from [-1,1], I label the score as following:
  - score >= 0.6: 5
  - 0.6 > score >= 0.2: 4
  - 0.2 > score >= -0.2: 3
  - -0.2 > score >= -0.4: 2
  - -1 >= score: 1

* Lastly, I will add two more features by determining the string character's length and word count.

### [EDA](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_EDA.py)

* Univariate analysis on target variable ("rating"): I first look at the count of each rating to see if there is any inbalance. From the bar chart in figure 2, ratings from "1" - "4" is in a close range (from 400 - 750 counts range), but rating of "5" has over 2600 counts. Although it is quite substantial, in term of business wise, there are many more customer that are very satisfied with the product than those that are not. We can say that, overall, customers are happy with their purchase.

<p align="center">
  <img width="500" height="300" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig2.png">
</p>

* Multivariate analysis on target variable ("rating"): next I add time as another element to the analysis. I am wondering whether time has any effect on the customers' rating: maybe the product was bad and people gave negative feedback then the manufacturer improved it and people liked it more or vice versa? Based on the figure below, it turned out that the trend is quite the same. From 2013 to 2017, the rating count is ranked that "1" has the lowest count and the higher the rating the higher the count. It is not until 2028 to 2020 that the count for "1" increases in terms of count and proportion of the rating for the year and is ranked as the second highest count rank. We do not have enough information to conclude that, in gerneral, customers are not liking the product anymore. But these are very good data for manufactor to start taking into consideration.

<p align="center">
  <img width="900" height="400" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig3.png">
</p>

* Next on the list are "rating" vs. "character_len" and "rating" vs. "word_count": because both box plots have very similar trend so I will analyze them as one. My initial assumption was that there would be a semi-clear trend of the higher the rating, the less words or character length the review has. Although there would be outliers (say a customer really loves the product and ending up writing paragraphs about it), my assumption is based on the fact that customer is more likely to criticize more when they are dissatisifed with the purchase. Two graphs below prove my assumption, but the trend is not as clear as I imagine it would be: the "5" rating's count range and median is smaller than the others, but the differences are not too significant. Maybe the result would be better if I graph these varaibles before cleanning the review text.

<p align="center">
  <img width="800" height="500" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig4_5.png">
</p>

* My last multivariate analysis is to see how accurate is the VADER Sentiment in column "predict_sentiment" by comparing its and the "rating" count. Based on the figure below, it does not predict as well as I would hope for. From "3" to "5" rating, the differences between the two counts are not as big of a gap. But it does pretty badly when trying to predict the sentiment of the "1" and "2" ratings. I can conclude that the VADER Sentiment might be over-rating the sentiment, so I need to expect seeing many reviews where they are rated negatively but the "predicted_sentiment" variable incorrectly states that it is positive.

<p align="center">
  <img width="500" height="400" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig6.png">
</p>

* After analyzing the variables, I want to have a closer look at the cleanned review text itself. I create a word cloud of all the review text for this product in figure 6. This has a very good representation of what words are being repeated the most by having it in different font sizes (the bigger the fonts, the higher the counts). Just by looking at the picture itself, beside the obvious, here are some observation:
  - I can see many text from "5" star ratings: "five star", "perfect", "buy", "great work", etc. 
  - I can also get a sense of what most of the reviews are about: "size", "quality", "durable", "fit". 
  - Negative review's texts are also present in the word cloud: "run small", "waist size", "tight", "return", etc.

<p align="center">
  <img width="900" height="500" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig7.png">
</p>

* Word cloud might be a great tool to have a general understanding of the text, but I can analyze it better if I look specifically at the word frequency of each ratings. Below are frequncey graphs of top 20 words from the five ratings. Here are some observations:
  - Figure 7a-7c: I grouped these three graphs together because they have similar trend. Just by looking at the top 5 words with highest requency ("size", "pant", "small", "waist", "fit"), I can already see that customers are not satisfied with the product is not happy about sizing the most. It is not a suprised to see many people would return the product because the word "return" is also in the graph.
  - Figure 7d: this is where we see the turning point the clearest: this graph has more positive words ranked top of the graph. Interestingly, the word "small" is still present in top 10.
  - Figure 7e: this graph is filled completely with only neutral to positive words. But also interesting to note, although reviews do seem to complement the fitting of the product, but they seem to be more postive than previous ratings.
  
<img width="390" height="350" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig8a.png"> <img width="375" height="350" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig8b.png">
<img width="390" height="350" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig8c.png"> <img width="375" height="350" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig8d.png">

<p align="center">
  <img width="390" height="350" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig8e.png">
</p>

* For my final analysis, I implement Latent Dirichlet Allocation (LDA) topic modeling. Although this step may seems repetitive, because the outcome might just be similar to what I have analyzed, I want to see how accurate is this unsupervised learning approach in identifying topics that are being talked about in the reviews' text. It turns out that LDA topic modeling identifies the topics quite well.

<img width="390" height="450" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig9a.png"> <img width="390" height="450" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig9b.png">

* This makes me wonder about the product sizing being polarized. Although different people have different shapes and sizes might be a good guess, but it does not align with the number of dissatisfied reviews. So, my assumption has to do with different sizes of the product might cause these criticisms: that is some product sizes might be scaled disproportionately.

### [Model Building](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_ModelBuilding.py)

* Perform vectorization on review texts: the process of converting words into numbers so that the machine can understand. For this project, I use Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams (bi-gram) technique. I have tried it with different ngrams (monogram, bi-gram, tri-gram) but it seems that bi-gram help to produce the best result.

* After removing unnecessary columns ("customer_id", "customer_name", "review_header", "review_body", "review_txt", "review_cleaned", I split the data to training (80%) and testing (20%) sets in a stratify fashion: stratas are the different ratings. The reason for this there is a lot more "5" stars rating compares to others. By splitting it in stratify fashion, I can eliminate the bias of having an over-whelmed majority of that rating.

* Because the data has many different ranges of value (such as character_len and those text vectorization columns), I rescale those values into a 0 to 1 range using the function `MinMaxScaler`. This helps the model produce a more accurate result.

* The first model I apply for this data set is the _Logistic Regression_. Although the model is known more for its binary classification, not quite what I want to predict (which is 5 different ratings), I want to apply it in this project to see how it does compare to other more advance ones. By having the "multi_class" parameter as "multinomial"(which uses the cross-entropy loss), I can apply it for my multiclass case. I also test it with different "C" (inverse of regularization strength) of values 0.01, 0.05, 0.25, 0.5, 1 to see which one gives the highest accuracy. Though this is a more simple model compare to the rest, so I expected the accuracy will not be as high.

```python
for x in [0.01, 0.05, 0.25, 0.5, 1]:
    log_reg = LogisticRegression(C=x, solver='lbfgs', multi_class='multinomial', random_state=1).fit(X_train, y_train)
    print ("Logistics regression accuracy (tfidf) for ", x, ":", accuracy_score(y_train, log_reg.predict(X_train)))

log_reg = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', random_state=1).fit(X_train, y_train)
print ("Logistics regression accuracy (tfidf) for C=1:", accuracy_score(y_train, log_reg.predict(X_train)))
```
```python
Out[2]: 0.9872372372372372
```

* I apply the _Gaussian Naive Bayes Classifier_, a variant of Naive Bayes, for the next model comparison. Since this algorithm has a different approach in building up a simple model (by assuming data is described by a normal distribution), I am curious to see how this simple supervised learning algorithms perform compares to others.

```python
naive_bayes = GaussianNB().fit(X_train, y_train)
print ("Naive Bayes accuracy: ", accuracy_score(y_train, naive_bayes.predict(X_train)))
```
```python
Out[4]: 0.9204204204204204
```

* Another supervised learning algorithm is the _random forest classifier_. But unlike the two algorithms above, it uses decision trees combination and voting mechanism to perform its prediction. Because the _random forest classifier_ is not biased and more stable when get fed with new data (opposite from Logistics Regression), I want to test out this algorithm after being concerned the algorithms above are both overfitted by producing such high accuracy scores (above 90%).

```python
randomfor_reg = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
print ("Random forest classifier accuracy: ", accuracy_score(y_train, randomfor_reg.predict(X_train)))
```
```python
Out[6]: 0.5367867867867868
```

* An algorithm that uses a similarity measure of data points to perform classification is _K-Nearest Neighbors Classification_. This is a good algorithm to use if the data is not linearly separable.

```python
k_neighbor = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
print ("K-Nearest neighbor accuracy: ", accuracy_score(y_train, k_neighbor.predict(X_train)))
```
```python
Out[8]: 0.5535535535535535
```

* Last but not least, I use the _Support vector machines (SVMs)_ as the last model to perform classification and compare the result. Because SVMs have been reported to work better for text classification and very effective in high dimensional spaces, I expect this to return relatively higher accuracy compares to the rest.

```python
supportvector = svm.SVC(decision_function_shape="ovo", random_state=1).fit(X_train, y_train)
print ("SVM accuracy: ", accuracy_score(y_train, supportvector.predict(X_train)))
```
```python
Out[10]: 0.9109109109109109
```

### Overall Model Performance

The accuracy scores of the training data set from the five models above suprise me. Specifically, the three models that give the highest scores are: Logistic Regression (98.72%), Gaussian Naive Bayes Classifier (92.04%), and Support vector machines (91.09%). My initial thought was that these models is overfitted. I would not be concerned if the two most simple classifier (Logistic and Gaussian Naive Bayes) would be overfitted, but SVMs should not be.

```python
log_reg_test = log_reg.predict(X_test)
naive_bayes_test = naive_bayes.predict(X_test)
randomfor_reg_test = randomfor_reg.predict(X_test)
k_neighbor_test = k_neighbor.predict(X_test)
supportvector_test = supportvector.predict(X_test)

print('Log regression: ', accuracy_score(y_test, log_reg_test))
print('Naive Bayes: ', accuracy_score(y_test, naive_bayes_test))
print('Random forest regression: ', accuracy_score(y_test, randomfor_reg_test))
print('K-nearest neighbor: ', accuracy_score(y_test, k_neighbor_test))
print('Support vector machines: ', accuracy_score(y_test, supportvector_test))
```

```
Log regression:  0.6396396396396397
Naive Bayes:  0.4244244244244244
Random forest regression:  0.5375375375375375
K-nearest neighbor:  0.5545545545545546
Support vector machines:  0.6046046046046046
```

## Conclusion



## Author

* **Chi Lam**, _student_ at Michigan State University - [chilam27](https://github.com/chilam27)

## Acknowledgments

[Abhinav Arora. "How to increase the model accuracy of logistic regression in Scikit python?" #138160970. 28 June 2016. Forum post.](https://stackoverflow.com/a/38083189/138160970)

[Bansal, S. (2016, August 24). Beginners Guide to Topic Modeling in Python and Feature Selection.](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)

[BhandariI, A. (2020, April 03). Feature Scaling: Standardization Vs Normalization.](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)

[Kub, A. (2019, January 24). Sentiment Analysis with Python (Part 2).](https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a)

[Loukas, S. (2020, June 14). ROC Curve Explained using a COVID-19 hypothetical example: Binary &amp; Multi-Class Classification...](https://towardsdatascience.com/roc-curve-explained-using-a-covid-19-hypothetical-example-binary-multi-class-classification-bab188ea869c)

[N, L. (2019, June 19). Sentiment Analysis of IMDB Movie Reviews.](https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/notebook)

[Rai, A. (2019, January 23). Python: Sentiment Analysis using VADER.](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/)

[Yann Dubois. "Expanding English language contractions in Python" #13816097. 03 November 2017. Forum post.](https://stackoverflow.com/a/47091490/13816097)
