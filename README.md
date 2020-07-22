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
  <img width="600" height="400" src="https://github.com/chilam27/Improved_Product_Review_System/blob/master/readme_image/fig1.png">
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



### [Model Building](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_ModelBuilding.py)



### Overall Model Performance



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
