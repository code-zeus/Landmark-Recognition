#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:58:23 2019

@author: kartikey
"""

import numpy as np
import pandas as pd
import urllib.request as ur
import cv2

def conv(mag,angle,b,a) :
	mat=np.zeros(9)
	for i in range(17*a,17*a+16):
		for j in range(17*b,17*b+16):
			m=0
			if(angle[i][j]>=180):
				m=angle[i][j]%180
			x=m%20
			y=m/20
			z=int(y)
			p=mag[i][j]*((20-x)/20)
			mat[z]+=p
			if(y>8):
				mat[1]+=mag[i][j]-p
			else:
				mat[z+1]+=mag[i][j]-p
	return mat


df=pd.read_csv("/home/kartikey/Desktop/Landmark/landmarks.csv")
df1=df["URL"]
for i in range(1):
	file_name='image.jpg'
	ur.urlretrieve(df1[0],file_name)
	img = cv2.imread(file_name,0)
	newimg= cv2.resize(img,(102,255))
	cv2.imshow('image',newimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#print(newimg)
	newimg = np.float32(newimg) / 255.0 
	gx = cv2.Sobel(newimg, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(newimg, cv2.CV_32F, 0, 1, ksize=1)
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	for m in range(0,16):
		for n in range(0,16):
			print(mag[m][n], end = ' ' )
		print('\n')
	bin_hist=np.zeros((15,6,9))
	for k in range(0,14):
		for j in range(0,5):
			bin_hist[k][j]=conv(mag,angle,j,i)
	

