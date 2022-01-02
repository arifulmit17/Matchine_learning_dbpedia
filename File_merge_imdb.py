# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:22:48 2022

@author: User
"""

import os
files = []
count=0
merge_file='E:\\aclImdb\\imdb_all.txt'
for f in os.scandir(path='E:\\aclImdb\\test\\neg'):
    
        if f.is_file():
          #print('it is file')
          files.append(f.path)
          count=count+1
          
 
#print(files)
print (count)
with open(merge_file, 'w' ) as allfile:
  for fp in files:
    with open(fp) as infile:
       allfile.write(infile.read())
    allfile.write('\n')
    
