# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:09:14 2021

@author: User
"""


import pandas as pd
#import preprocessing as pp
import bag_of_words as bow
from sklearn import model_selection,preprocessing
import Naive_Bayes as nb
import tf_idf as tfi
my_tags=['Company',
'EducationalInstitution',
'Artist',
'Athlete',
'OfficeHolder',
'MeanOfTransportation',
'Building',
'NaturalPlace',
'Village',
'Animal',
'Plant',
'Album',
'Film',
'WrittenWork'
]
train_data=pd.read_csv('Dbpedia_train_2.csv')
train_data=train_data[pd.notnull(train_data['Class'])]
print(train_data['Description'].apply(lambda x: len(x.split(' '))).sum())
X = train_data.Description
y = train_data.Class
print(y)
#train_data['text']=X

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X,y,test_size=0.10, random_state = 0)
print (train_x.shape)
print (valid_x.shape)
print (train_y.shape)
print (valid_y.shape)
train=[]
train=bow.bag_of_words(train_data,X,train_x, valid_x)
xtrain_count=train[0]
xvalid_count=train[1]
#print(xvalid_count)

tf_idf=[]
tf_idf=tfi.tf_idf(train_data,X,train_x, valid_x)
xtrain_tfidf=tf_idf[0]
xvalid_tfidf=tf_idf[1]
xtrain_tfidf_ngram=tf_idf[2]
xvalid_tfidf_ngram=tf_idf[3]
xtrain_tfidf_ngram_chars=tf_idf[4]
xvalid_tfidf_ngram_chars=tf_idf[5]
nb.Naive_Bayes(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars)