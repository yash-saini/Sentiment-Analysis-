# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:29:22 2019

@author: YASH SAINI
"""
import re
import tweepy
import csv
import nltk
import pandas as pd
import string
import math
from tweepy import OAuthHandler 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')
#nltk.download('vader_lexicon')

class Twitter_Main():
    def __init__(self):
        #input your credentials here
        #Generate a twitter developer's account for getting the following.
        ''' The code will not work unless the keys are writter.'''
        consumer_key = '#'
        consumer_secret = '#'
        access_token = '#'
        access_token_secret = '#'
        try:
            self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth,wait_on_rate_limit=True)
        except:
            print("\nInvalid Authentication")
        
        #Stopwords file opening
        with open('stopwords_file.txt', 'r') as f:
            k=f.readlines()
        self.stoplist=[k[i][:-1] for i in xrange(len(k))]
    
    def removePostfix(self,argWord):
        #stemming
        leaves = ["s", "es", "ed", "er", "ly", "ing"]
        for leaf in leaves:
            if argWord[-len(leaf):] == leaf:
                return argWord[:-len(leaf)]
            
        return argWord
        
            
    def preprocessing(self,tweet):
        
        #Convert to lower case
        tweet = tweet.lower()
        #Removing www.* or https?://*  or @*
        tweet = re.sub('www\.[^\s]+',' ',tweet)
        tweet=re.sub("https?://[^\s]+",' ',tweet)
        tweet= re.sub("@[^\s]+",' ' , tweet)
        #Remove rt Retweet
        tweet=re.sub("rt",' ',tweet)
        #Remove #word 
        tweet = re.sub(r'#([^\s]+)', ' ', tweet)
        #Remove punctuations
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        tweet=regex.sub('', tweet)
        #Only English words
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)
        #tokenize or split in words
        tweet = tweet.split()
        #Remove duplicate words and extra spaces 
        tweet=list(set(tweet))
        if ' ' in tweet:
            tweet.remove(' ')

            
        tweet1=[]
        
        #Lemmatizing
        wordnet_lemmatizer = WordNetLemmatizer()
        for i in xrange(len(tweet)):
            if tweet[i] not in self.stoplist:
                #Stemming
                c=self.removePostfix(tweet[i])
                t=wordnet_lemmatizer.lemmatize(c)
                tweet1.append(t)
                    
         
        tweet1 = ' '.join(tweet1)
        return tweet1 
    
   
    
    def get_tweets(self, query, count = 500): 
        ''' 
        Function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
  
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q = query, count = count) 
  
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  

                parsed_tweet[self.preprocessing(tweet.text)] = self.get_tweet_sentiment(tweet.text) 
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
  
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # error (if any) 
            print("Error : " + str(e)) 
        
    def get_tweet_sentiment(self, tweet): 
        ''' 
        Function assigns a sentiment score to each tweet given as an input
        Uses VADER sentiment analyzer technique of NLTK
        
        '''
        s=SentimentIntensityAnalyzer()
        score = s.polarity_scores(tweet)
        # set sentiment 
        if score['compound']>=0.05: 
            return 'positive'
        elif    -0.05< score['compound']<0.05 : 
            return 'neutral'
        else: 
            return 'negative'

 
# creating object of Twitter_Main Class 
api = Twitter_Main() 
# calling function to get tweets 
k1=raw_input("Enter the query:")
tweets = api.get_tweets(query = k1, count = 500) 
print("\n---Process is over\n")
k=raw_input("Enter the name of the file \n to be saved with tweets and sentiments:")
k=k+'.csv'
''' Store the tweets and the sentiments'''
with open(k, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('Tweets', 'Sentiment'))
    for l in tweets:
        for key, value in l.items():
            if type(key) == float and np.isnan(key):
                pass
            else:
                writer.writerow([key, value])
            
