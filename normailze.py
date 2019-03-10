# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:44:32 2019

@author: pushkarpathak
"""

import numpy as np

def normalize(temp):
    sum=0
    for i in range(0,temp.size-1):
        sum+=temp[i]**2
    sum=sum**(0.5)
    for i in range(0,temp.size-1):
        temp[i]=temp[i]/sum
    return temp

for i in range(0,4):
    for j in range(0,13):
            temp=np.concatenate((bin_hist[i][j],bin_hist[i+1][j],bin_hist[i][j+1],bin_hist[i+1][j+1]))
            hog_descriptor=np.concatenate((hog_descriptor,normalize(temp)))
                