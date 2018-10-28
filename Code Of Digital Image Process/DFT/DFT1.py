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
f1Value = np.abs(f1)
f1Angle = np.angle(f1)

iImgV = np.zeros(img1.shape, dtype = complex)
#angleImg
iImgV.real = np.array(np.cos(f1Angle))
iImgV.imag = np.array(np.sin(f1Angle))
iImgAngle = np.fft.ifft2(iImgV)
#valueImg
iImgV.real = np.array(f1Value*np.cos(0))
iImgV.imag = np.array(f1Value*np.sin(0))
iImgValue = np.fft.ifft2(iImgV)
#iImgTotal
iImgV.real = np.array(f1Value*np.cos(f1Angle))
iImgV.imag = np.array(f1Value*np.sin(f1Angle))
iImgTotal = np.fft.ifft2(iImgV)

#fshift1 = np.fft.fftshift(f1)
#fshiftAngle1 = np.angle(f1)
#f2shiftAngle1 = np.fft.ifftshift(fshiftAngle1)

#s1 = (np.abs(f1))
#siuint64=np.uint64(np.abs(f1))
#s1Uint8 = np.uint8(s1)
#s1_real = s1*np.cos(fshiftAngle1)
#s1_imag = s1*np.sin(fshiftAngle1)
#s2 = np.zeros(img1.shape,dtype=complex) 
#s2.real = s1#np.array(s1_real)
#s2.imag = 0#np.array(s1_imag)

#imgBack = np.fft.ifft2(s2)


displayImg(151,img1,'f1')
displayImg(152,np.uint8(f1Value),'b')
displayImg(153,np.abs(iImgAngle),'c')
displayImg(154,np.abs(iImgValue),'d')
displayImg(155,np.abs(iImgTotal),'e')
plt.show()

#opencv
#dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#dft_shift = np.fft.fftshift(dft)

#s1 = 1 + np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

#plt.subplot(221),plt.imshow(img,'gray'),plt.title('a')
#plt.subplot(222),plt.imshow(cv2.magnitude(dft[:,:,0], dft[:,:,1]),'gray'),plt.title('b')
#plt.subplot(223),plt.imshow(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]),'gray'),plt.title('c')
#plt.subplot(224),plt.imshow(s1,'gray'),plt.title('d')

