# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:00:07 2022

@author: User
"""

import os
files = []
for f in os.scandir(path='E:\\aclImdb'):
    if f.is_dir():
        print('it is folder')
        for g in os.scandir(f.path):    
          if g.is_dir():
             print('it is another folder')   
          elif g.is_file():
             print('it is file')
             files.append(g.path)
        
    elif f.is_file():
        print('it is file')
        files.append(f.path)
 
print(files)