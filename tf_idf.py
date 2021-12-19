import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def tf_idf(train_data,X,train_x, valid_x):
          train_data['text']=X
          # word level tf-idf
          tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
          tfidf_vect.fit(train_data['text'])
          xtrain_tfidf =  tfidf_vect.transform(train_x)
          xvalid_tfidf =  tfidf_vect.transform(valid_x)

          # ngram level tf-idf 
          tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
          tfidf_vect_ngram.fit(train_data['text'])
          xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
          xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
          tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
          tfidf_vect_ngram_chars.fit(train_data['text'])
          xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
          xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
          return xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars

