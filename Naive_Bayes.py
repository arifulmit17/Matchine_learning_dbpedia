import modeling as md
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def Naive_Bayes(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars):
     from sklearn import naive_bayes 
     # Naive Bayes on Bag of words
     predictions = md.train_model(naive_bayes.MultinomialNB(),  xtrain_count, train_y,xvalid_count)
     print ("MultinomialNB, Bag of words: ", accuracy_score(predictions, valid_y))
     cm = confusion_matrix(valid_y, predictions)
     print (cm)
     print(classification_report(valid_y,predictions,target_names=my_tags))
     # Naive Bayes on Word Level TF IDF Vectors
     predictions = md.train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
     print ("MultinomialNB, WordLevel TF-IDF: ",accuracy_score(predictions, valid_y))
     cm = confusion_matrix(valid_y, predictions)
     print (cm)
     print(classification_report(valid_y,predictions,target_names=my_tags))
     # Naive Bayes on Ngram Level TF IDF Vectors
     predictions = md.train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
     print ("MultinomialNB, N-Gram TF-IDF: ", accuracy_score(predictions, valid_y))
     cm = confusion_matrix(valid_y, predictions)
     print (cm)
     print(classification_report(valid_y,predictions,target_names=my_tags))
     # Naive Bayes on Character Level TF IDF Vectors
     predictions = md.train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
     print ("MultinomialNB, CharLevel TF-IDF: ",accuracy_score(predictions, valid_y))
     cm = confusion_matrix(valid_y, predictions)
     print (cm)
     print(classification_report(valid_y,predictions,target_names=my_tags))

