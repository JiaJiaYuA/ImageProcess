# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:20:34 2018

@author: JIAJIAXIANG
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def readImage(fileName):
    return cv2.imread(fileName, 0)
def writeImage(path,img):
    cv2.imwrite(path,np.uint8(img))
def displayImg(index,img,title):
    plt.subplot(index),plt.imshow(img,'gray'),plt.title(title)

img1 = readImage('woman.tif')

#matlab
f1 = np.fft.fft2(img1)
#fshift1 = np.fft.fftshift(f1)
fshiftAngle1 = np.angle(f1)
#f2shiftAngle1 = np.fft.ifftshift(fshiftAngle1)

s1 = (np.abs(f1))
siuint64=np.uint64(np.abs(f1))
s1Uint8 = np.uint8(s1)
#s1_real = s1*np.cos(fshiftAngle1)
#s1_imag = s1*np.sin(fshiftAngle1)
s2 = np.zeros(img1.shape,dtype=complex) 
s2.real = s1#np.array(s1_real)
s2.imag = 0#np.array(s1_imag)

imgBack = np.fft.ifft2(s2)


displayImg(121,img1,'a')
displayImg(122,s1,'b')
#displayImg(122,np.abs(imgBack),'b')
plt.show()

#opencv
#dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#dft_shift = np.fft.fftshift(dft)

#s1 = 1 + np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

#plt.subplot(221),plt.imshow(img,'gray'),plt.title('a')
#plt.subplot(222),plt.imshow(cv2.magnitude(dft[:,:,0], dft[:,:,1]),'gray'),plt.title('b')
#plt.subplot(223),plt.imshow(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]),'gray'),plt.title('c')
#plt.subplot(224),plt.imshow(s1,'gray'),plt.title('d')

