# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#导入cv模块
import cv2
import numpy as np
from matplotlib import pyplot as plt

#伽马变换
def gamma_trans(img,gamma,c):
    gamma_table = [np.power(x,gamma)*c for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(imgSrcMulty, gamma_table)

kernel = np.array([ [-1, -1, -1],
                   [-1, 8, -1],
                   [-1,-1, -1] ])
knlMeanFilter = np.ones((5,5),np.float32)/25

imgSrc = cv2.imread("bones.tif", 0)
#filter为卷积运算
laplacian = cv2.filter2D(imgSrc, -1, kernel)

sobelx16S = cv2.Sobel(imgSrc,cv2.CV_16S,1,0,ksize=3)
sobely16S = cv2.Sobel(imgSrc,cv2.CV_16S,0,1,ksize=3)
abs_sobelx16U = np.absolute(sobelx16S)
abs_sobely16U = np.absolute(sobely16S)
abs_sobelxy = abs_sobelx16U + abs_sobely16U

imgSobel = np.uint8(abs_sobelxy*0.5)
#imgSobel = cv2.convertScaleAbs(abs_sobelxy, alpha = (255.0)/np.max(abs_sobelxy))

imgMean = cv2.filter2D(imgSobel, -1, knlMeanFilter)

imgSrcPlusLaplacian = imgSrc + laplacian
imgMulltyTmp = np.multiply(imgSrcPlusLaplacian.astype(np.int16), imgMean.astype(np.int16))
imgMullty = cv2.convertScaleAbs(imgMulltyTmp, alpha=(255.0/65535.0))
imgSrcMulty = cv2.add(imgSrc, imgMullty)
imgFinish = gamma_trans(imgSrcMulty, 0.5, 1)

plt.subplot(2,4,1), plt.imshow(imgSrc, cmap = 'gray')
plt.title("Original")
plt.subplot(2,4,2), plt.imshow(imgSrcPlusLaplacian, cmap = 'gray')
plt.title("Laplacian")
plt.subplot(2,4,3), plt.imshow(imgSobel, cmap = 'gray')
plt.title("Sobel")
plt.subplot(2,4,4), plt.imshow(imgMean, cmap = 'gray')
plt.title("Mean")
plt.subplot(2,4,5), plt.imshow(imgMullty, cmap = 'gray')
plt.title("Multity")
plt.subplot(2,4,6), plt.imshow(imgSrcMulty, cmap = 'gray')
plt.title("SrcMultity")
plt.subplot(2,4,7), plt.imshow(imgFinish, cmap = 'gray')
plt.title("Finished")

plt.show()