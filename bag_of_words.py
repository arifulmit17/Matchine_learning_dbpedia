from sklearn.feature_extraction.text import  CountVectorizer
def bag_of_words(train_data,X,train_x, valid_x):
    train_data['text']=X
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train_data['text'])
    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    return xtrain_count,xvalid_count
