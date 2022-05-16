from msilib.schema import Class
from multiprocessing import Value
from tkinter import Y
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
labels=['Company',
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

#train_data=pd.read_csv('dbpedia_train_3.csv', encoding = 'latin1')
train_data_chunk=pd.read_csv('dbpedia_train_3.csv', encoding = 'iso-8859-1',chunksize=5000,dtype={'Class':int,'Value':str,'Description':str})
#train_data=pd.read_csv('dbpedia_train_3.csv', encoding = 'iso-8859-1')
for i in train_data_chunk:
 print(i.shape)
 train_data=i
 
 print(type(train_data))
#train_data=pd.read_csv('dbpedia_train_3.csv', encoding = 'cp1252')
#train_data=pd.read_csv('dbpedia_train.csv',encoding = 'latin1')
#train_data=pd.read_csv('Dbpedia_train_2.csv')
 print(train_data.head())
#test_data=pd.read_csv('dbpedia_test_2.csv',encoding = 'latin1')
 test_data_chunk=pd.read_csv('dbpedia_test_2.csv',encoding = 'iso-8859-1',chunksize=200)
 for i in test_data_chunk:
  print(i.shape)
  test_data=i
#test_data=pd.read_csv('dbpedia_test_2.csv',encoding = 'cp1252')

#test_data=pd.read_csv('Dbpedia_test.csv')
 lens = train_data.Description.str.len()
 print(lens.mean(), lens.std(), lens.max())
#print(test_data.head())
#print(train_data.describe())
 print(len(train_data))
 import re, string
 re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
 def tokenize(s): return re_tok.sub(r' \1 ', s).split()

 n = train_data.shape[0]
#print(train_data['Description'])
 vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
 trn_term_doc = vec.fit_transform(train_data['Description'])
#print(trn_term_doc)
#print(test_data.head())
 print(len(test_data))
 test_term_doc=vec.transform(test_data['Description'])

 def pr(y_i, y):
    p = x[y==y_i].sum(0)
    #print(p)
    return (p+1) / ((y==y_i).sum()+1)

 x = trn_term_doc
 test_x=test_term_doc

 def get_mdl(y):
    #print(y)
    y = y
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

 preds = np.zeros((len(test_data), len(labels)))


 for i,j in enumerate(labels):
    print('fit', j)
    m, r = get_mdl(train_data['Class'])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
 print(preds)

#submid = pd.DataFrame({'id': subm["id"]})
 submission = pd.DataFrame(preds, columns = labels)
 submission.to_csv('submission_2.csv', index=False)