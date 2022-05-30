def lenshist(lens):
 import matplotlib.pyplot as plt
 import scipy.stats as stat
 fit=stat.norm.pdf(lens,lens.mean(), lens.std())
 plt.figure(figsize=(10,10))
 

 #plt.subplot(222)
 plt.plot(lens,fit,'-o',scalex=10)
 plt.show()