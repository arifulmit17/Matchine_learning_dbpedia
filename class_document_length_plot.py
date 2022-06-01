def lenhist(lens):
 import matplotlib.pyplot as plt
 plt.figure(figsize=(10,10))
 plt.xlabel('Length of a document', color='white')
 plt.ylabel('number of documents',color='white')
 
 #plt.xlim([1,2500])
 plt.legend()
 #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
 #plt.hist(lens,density=True,stacked=True)
 #plt.subplot(223)
 plt.hist(lens)
 plt.show()