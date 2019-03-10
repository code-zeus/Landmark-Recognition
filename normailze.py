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

                