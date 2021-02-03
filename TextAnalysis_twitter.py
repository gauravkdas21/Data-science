# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:53:09 2019

@author: Gaurav.Das
"""

# Twitter Sentiment Analysis 

#Table of Contents
#Understand the Problem Statement
#Tweets Preprocessing and Cleaning
#Story Generation and Visualization from Tweets
#Extracting Features from Cleaned Tweets
#Model Building: Sentiment Analysis

# 1. Understand the Problem Statement

#The objective of this task is to detect hate speech in tweets. For the sake of simplicity, 
#we say a tweet contains hate speech if it has a racist or sexist sentiment associated with 
#it. So, the task is to classify racist or sexist tweets from other tweets.

#Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet 
#is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, your objective is 
#to predict the labels on the given test dataset.

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

train  = pd.read_csv(r'data\\twitter\\train_E6oV3lV.csv')
test = pd.read_csv(r'data\\twitter\\test_tweets_anuFYb8.csv')
#Let’s check the first few rows of the train dataset.

train.head()

#The data has 3 columns id, label, and tweet. label is the binary target variable and tweet
# contains the tweets that we will clean and preprocess.

#Initial data cleaning requirements that we can think of after looking at the top 5 records:

#The Twitter handles are already masked as @user due to privacy concerns. So, these Twitter
# handles are hardly giving any information about the nature of the tweet.
#We can also think of getting rid of the punctuations, numbers and even special characters 
#since they wouldn’t help in differentiating different kinds of tweets.
#Most of the smaller words do not add much value. For example, ‘pdx’, ‘his’, ‘all’. So, we 
#will try to remove them as well from our data.
#Once we have executed the above three steps, we can split every tweet into individual words
# or tokens which is an essential step in any NLP task.
#In the 4th tweet, there is a word ‘love’. We might also have terms like loves, loving, lovable,
# etc. in the rest of the data. These terms are often used in the same context. If we can reduce
# them to their root word, which is ‘love’, then we can reduce the total number of unique words
# in our data without losing a significant amount of information.
 

#A) Removing Twitter Handles (@user)
#As mentioned above, the tweets contain lots of twitter handles (@user), that is how a Twitter
# user acknowledged on Twitter. We will remove all these twitter handles from the data as they
# don’t convey much information.

#For our convenience, let’s first combine train and test set. This saves the trouble of performing
# the same steps twice on test and train.

combi = train.append(test, ignore_index=True)
#Given below is a user-defined function to remove unwanted text patterns from the tweets. It 
#takes two arguments, one is the original string of text and the other is the pattern of text
#that we want to remove from the string. The function returns the same input string but without
# the given pattern. We will use this function to remove the pattern ‘@user’ from all the tweets
# in our data.

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt   

 
#Now let’s create a new column tidy_tweet, it will contain the cleaned and processed tweets. 
#Note that we have passed “@[\w]*” as the pattern to the remove_pattern function. It is actually
# a regular expression which will pick any word starting with ‘@’.

# remove twitter handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")


#B) Removing Punctuations, Numbers, and Special Characters
#As discussed, punctuations, numbers and special characters do not help much. It is better to 
#remove them from the text just as we removed the twitter handles. Here we will replace 
#everything except characters and hashtags with spaces.

# remove special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
 

#C) Removing Short Words
#We have to be a little careful here in selecting the length of the words which we want to 
#remove. So, I have decided to remove all the words having length 3 or less. For example, 
#terms like “hmm”, “oh” are of very little use. It is better to get rid of them.

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#Let’s take another look at the first few rows of the combined dataframe.

combi.head()

#You can see the difference between the raw tweets and the cleaned tweets (tidy_tweet) quite
# clearly. Only the important words in the tweets have been retained and the noise (numbers,
# punctuations, and special characters) has been removed.

 

#D) Tokenization
#Now we will tokenize all the cleaned tweets in our dataset. Tokens are individual terms 
#or words, and tokenization is the process of splitting a string of text into tokens.

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

#E) Stemming
#Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) 
#from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing”
# are the different variations of the word – “play”.

#from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()

#Now let’s stitch these tokens back together.

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet


#3. Story Generation and Visualization from Tweets
#In this section, we will explore the cleaned tweets text. Exploring and visualizing data, 
#no matter whether its text or any other data, is an essential step in gaining insights. 
#Do not limit yourself to only these methods told in this tutorial, feel free to explore 
#the data as much as possible.

#Before we begin exploration, we must think and ask questions related to the data in hand. 
#A few probable questions are as follows:

#What are the most common words in the entire dataset?
#What are the most common words in the dataset for negative and positive tweets, respectively?
#How many hashtags are there in a tweet?
#Which trends are associated with my dataset?
#Which trends are associated with either of the sentiments? Are they compatible with the sentiments?
 

#A) Understanding the common words used in the tweets: WordCloud
#Now I want to see how well the given sentiments are distributed across the train dataset. 
#One way to accomplish this task is by understanding the common words by plotting wordclouds.

#A wordcloud is a visualization wherein the most frequent words appear in large size and the 
#less frequent words appear in smaller sizes.

#Let’s visualize all the words our data using the wordcloud plot.

all_words = ' '.join([text for text in combi['tidy_tweet']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#We can see most of the words are positive or neutral. With happy and love being the 
#most frequent ones. It doesn’t give us any idea about the words associated with the 
#racist/sexist tweets. Hence, we will plot separate wordclouds for both the classes
#(racist/sexist or not) in our train data.

#B) Words in non racist/sexist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#We can see most of the words are positive or neutral. With happy, smile, and love being
# the most frequent ones. Hence, most of the frequent words are compatible with the sentiment
# which is non racist/sexists tweets. Similarly, we will plot the word cloud for the other
# sentiment. Expect to see negative, racist, and sexist terms.

#C) Racist/Sexist Tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#As we can clearly see, most of the words have negative connotations. So, it seems we have
# a pretty good text data to work on. Next we will use the hashtags/trends in our twitter data.

 
#D) Understanding the impact of Hashtags on tweets sentiment
#Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular 
#point in time. We should try to check whether these hashtags add any value to our sentiment
# analysis task, i.e., they help in distinguishing tweets into the different sentiments.

#For instance, given below is a tweet from our dataset:


#The tweet seems sexist in nature and the hashtags in the tweet convey the same feeling.

#We will store all the trend terms in two separate lists — one for non-racist/sexist tweets 
#and the other for racist/sexist tweets.

# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
#Now that we have prepared our lists of hashtags for both the sentiments, we can plot 
#the top n hashtags. So, first let’s check the hashtags in the non-racist/sexist tweets.

#Non-Racist/Sexist Tweets

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


#All these hashtags are positive and it makes sense. I am expecting negative terms in the
# plot of the second list. Let’s check the most frequent hashtags appearing in the 
#racist/sexist tweets.

#Racist/Sexist Tweets

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


#As expected, most of the terms are negative with a few neutral terms as well. So, it’s not 
#a bad idea to keep these hashtags in our data as they contain useful information. Next, we
# will try to extract features from the tokenized tweets.

 

#4. Extracting Features from Cleaned Tweets
#To analyze a preprocessed data, it needs to be converted into features. Depending upon the
# usage, text features can be constructed using assorted techniques – Bag-of-Words, TF-IDF,
# and Word Embeddings. In this article, we will be covering only Bag-of-Words and TF-IDF.


#Bag-of-Words Features
#Bag-of-Words is a method to represent text into numerical features. Consider a corpus
# (a collection of texts) called C of D documents {d1,d2…..dD} and N unique tokens extracted
# out of the corpus C. The N tokens (words) will form a list, and the size of the bag-of-words
# matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens
# in document D(i).

#Let us understand this using a simple example. Suppose we have only 2 document

#D1: He is a lazy boy. She is also lazy.

#D2: Smith is a lazy person.

#The list created would consist of all the unique tokens in the corpus C.

#= [‘He’,’She’,’lazy’,’boy’,’Smith’,’person’]

#Here, D=2, N=6

#The matrix M of size 2 X 6 will be represented as –



#Now the columns in the above matrix can be used as features to build a classification 
#model. Bag-of-Words features can be easily created using sklearn’s CountVectorizer 
#function. We will set the parameter max_features = 1000 to select only top 1000 terms
# ordered by term frequency across the corpus.

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
#TF-IDF Features
#This is another method which is based on the frequency method but it is different to 
#the bag-of-words approach in the sense that it takes into account, not just the occurrence
# of a word in a single document (or tweet) but in the entire corpus.

#TF-IDF works by penalizing the common words by assigning them lower weights while giving 
#importance to words which are rare in the entire corpus but appear in good numbers in 
#few documents.

#Let’s have a look at the important terms related to TF-IDF:

#TF = (Number of times term t appears in a document)/(Number of terms in the document)
#IDF = log(N/n), where, N is the number of documents and n is the number of documents a
# term t has appeared in.
#TF-IDF = TF*IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
 

#5. Model Building: Sentiment Analysis
#We are now done with all the pre-modeling stages required to get the data in the proper
# form and shape. Now we will be building predictive models on the dataset using the two
# feature set — Bag-of-Words and TF-IDF.

#We will use logistic regression to build the models. It predicts the probability of occurrence
# of an event by fitting data to a logit function.

#The following equation is used in Logistic Regression:
    

#A) Building model using Bag-of-Words features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 
#than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score
#Output: 0.53

#We trained the logistic regression model on the Bag-of-Words features and it gave us an 
#F1-score of 0.53 for the validation set. Now we will use this model to predict for the test data.

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv(r'data\\twitter\\sub_lreg_bow.csv', index=False) # writing data to a CSV file
#The public leaderboard F1 score is 0.567. Now we will again train a logistic regression
# model but this time on the TF-IDF features. Let’s see how it performs.

 

#B) Building model using TF-IDF features
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)
#Output: 0.544

#The validation score is 0.544 and the public leaderboard F1 score is 0.564. So, by using 
#the TF-IDF features, the validation score has improved and the public leaderboard score 
#is more or less the same.



