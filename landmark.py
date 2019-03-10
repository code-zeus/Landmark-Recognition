#Landmark Recognition

import numpy as np
from normailze import normalize
import pandas as pd
import urllib.request as ur
import cv2

def conv(mag,angle,b,a) :
	mat=np.zeros(9)
	for i in range(17*a,17*a+17):
		for j in range(17*b,17*b+17):
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
              

df=pd.read_csv(r"C:\Users\pushkarpathak\Desktop\Landmark-Recognition\landmarks.csv")
df1=df["URL"]
for i in range(2):
	file_name='image.jpg'
	ur.urlretrieve(df1[0],file_name)
	img = cv2.imread(file_name,0)
	newimg= cv2.resize(img,(102,255))
	cv2.imshow('image',newimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	newimg = np.float32(newimg) / 255.0 
	gx = cv2.Sobel(newimg, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(newimg, cv2.CV_32F, 0, 1, ksize=1)
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	bin_hist=np.zeros((15,6,9))
	for k in range(0,15):
		for j in range(0,6):
			bin_hist[k][j]=conv(mag,angle,j,i)   
            
            
            
hog_descriptor=np.zeros((0))
for i in range(0,14):
    for j in range(0,5):
        temp=np.concatenate((bin_hist[i][j],bin_hist[i+1][j],bin_hist[i][j+1],bin_hist[i+1][j+1]))
        hog_descriptor=np.concatenate((hog_descriptor,normalize(temp)))          
for i in range(hog_descriptor.size-1):
    print(hog_descriptor[i])

