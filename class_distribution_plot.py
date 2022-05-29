from ast import increment_lineno
import matplotlib


def vchist(vc):
 import matplotlib.pyplot as plt

 plt.xlabel('Classes ', color='white')
 plt.ylabel('number of documents per class',color='white')
 #plt.xlim([1,2500])
 plt.legend()
 #plt.subplot(221)
 #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
 #plt.hist(vc,label=['Company','EducationalInstitution','Artist','Athlete','OfficeHolder','MeanOfTransportation','Building','NaturalPlace','Village','Animal','Plant','Album','Film','WrittenWork'])
 
 vc.plot.hist(bins=14,figsize=(15,15))
 plt.show()