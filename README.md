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



### [Data Collection](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_DataCollection.py)



### [Text Preprocessing](https://github.com/chilam27/Improved_Product_Review_System/blob/master/P03_TextPreprocessing.py)



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
