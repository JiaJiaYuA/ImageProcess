# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:20:34 2018

@author: JIAJIAXIANG
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('rectangle.tif',0)

#matlab
#f = np.fft.fft2(img)
#fshift = np.fft.fftshift(f)
#s1 = 1 + np.log(np.abs(fshift))

#plt.subplot(221),plt.imshow(img,'gray'),plt.title('a')
#plt.subplot(222),plt.imshow(np.abs(f),'gray'),plt.title('b')
#plt.subplot(223),plt.imshow(np.abs(fshift),'gray'),plt.title('c')
#plt.subplot(224),plt.imshow(s1,'gray'),plt.title('d')

#opencv
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

s1 = np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

plt.subplot(221),plt.imshow(img,'gray'),plt.title('a')
plt.subplot(222),plt.imshow(cv2.magnitude(dft[:,:,0], dft[:,:,1]),'gray'),plt.title('b')
plt.subplot(223),plt.imshow(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]),'gray'),plt.title('c')
plt.subplot(224),plt.imshow(s1,'gray'),plt.title('d')

