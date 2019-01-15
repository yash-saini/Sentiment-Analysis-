# Sentiment-Analysis-
The project 'Sentiment Analysis of top colleges in India using Twitter Data',  extracts tweets from the twitter , preprocesses them , assigns sentiment scores and then the application of various supervised Machine Learning techniques on the data generated is performed and their accuracies are calculated.
The tweets are downloaded for three main colleges considered in this project( IIT-D, JNU, AIIMS) .These tweets are downloaded through the twitter api. These tweets are then preprocessed by performing the following steps:
1)Conversion to lowercase alphabets
2)Removal of www * or https?://*  or @* , hashtag words, stop words (i,am,is ...etc),  punctuations, non English words, duplicate words and extra spaces.
3)Stemming and lemmetization on the cleaned tweets is performed.
These cleaned tweets are then processed for the assignment of sentiments to each of them. The sentiment scores are assigned using the VADER approach of the nltk python package and the data is stored in a file.

Machine Learning on the dataset:-
The dataset is retrieved,  and divided in the training and test data using the hold-out method (7:3). Various ML techniques namely KNN,SVM (with rbf kernel) , Naive bayes and xgboost are used and their accuracies are compared.
