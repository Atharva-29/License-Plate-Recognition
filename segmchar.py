import numpy as np
import cv2 
import scipy.fftpack 

from keras.preprocessing.image import ImageDataGenerator as idg
from keras.models import Sequential as seq
from keras.layers import Conv2D
from keras.layers import MaxPooling2D as MaxP2D
from keras.layers import Dense, Flatten,Dropout
import os
import sys
import imghdr

image_width, image_height = 128,128


def imclearborder(imgBW, radius):

    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] 

    for idx in np.arange(len(contours)):
        cnt = contours[idx]
        for pt in cnt:

            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


def bwareaopen(imgBW, areaPixels):

    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
    return imgBWcopy


def extract_char(image_given):
    image_copy = image_given.copy()
    contours,hierarchy = cv2.findContours(image_copy, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    n=0

    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        roi = image_copy[y:y+h+10, x:x+w+7]
        # cv2.imshow('character: %d'%n,roi)
        roi=cv2.bitwise_not(roi)
        cv2.imwrite('/Users/atharvaajitagwekar/Desktop/LPR/%d.png'%n, roi)
        n = n+1

def create_model():
  alpha = seq()
  alpha.add(Conv2D(32,[3,3],strides=1,padding='valid',activation='relu',
                          input_shape=(image_width, image_height, 3)))
  alpha.add(Conv2D(32,[3,3],strides=1,padding='valid',activation='relu'))
  alpha.add(MaxP2D(pool_size=(2,2)))

  alpha.add(Conv2D(64,[3,3],strides=1,padding='valid',activation='relu'))
  alpha.add(MaxP2D(pool_size=(2,2)))

  alpha.add(Conv2D(64,[3,3],strides=1,padding='valid',activation='relu'))
  alpha.add(Conv2D(128,[3,3],strides=1,padding='valid',activation='relu'))
  alpha.add(MaxP2D(pool_size=(2,2)))


  alpha.add(Flatten())
  alpha.add(Dense(1024, activation='relu'))
  alpha.add(Dropout(0.4))
  alpha.add(Dense(36, activation='softmax'))
    
  return alpha

def ret_character(i):
  characters_list=['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  return characters_list[i]

def ret_pred(provided_image):

  img = cv2.imread(provided_image, 0)

  rows = img.shape[0]
  cols = img.shape[1]

  img = img[:, 15:cols-20]

  rows = img.shape[0]
  cols = img.shape[1]

  imgLog = np.log1p(np.array(img, dtype="float") / 255)

  M = 2*rows + 1
  N = 2*cols + 1
  sigma = 10
  (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
  centerX = np.ceil(N/2)
  centerY = np.ceil(M/2)
  gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

  Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
  Hhigh = 1 - Hlow

  HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
  HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

  If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
  Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
  Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

  gamma1 = 0.3
  gamma2 = 1.5
  Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

  Ihmf = np.expm1(Iout)
  Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
  Ihmf2 = np.array(255*Ihmf, dtype="uint8")

  Ithresh = Ihmf2 < 65
  Ithresh = 255*Ithresh.astype("uint8")

  Iclear = imclearborder(Ithresh, 5)

  Iopen = bwareaopen(Iclear, 120)

  # cv2.imshow('Original Image', img)
  # cv2.imshow('Homomorphic Filtered Result', Ihmf2)
  # cv2.imshow('Thresholded Result', Ithresh)
  # cv2.imshow('Opened Result', Iopen)
  cv2.imwrite('result.jpg',Iopen)
  
  extract_char(Iopen)

  image_width, image_height = 128,128
  license_number=[]

  for file in sorted(os.listdir("/Users/atharvaajitagwekar/Desktop/LPR/")):
    if imghdr.what("/Users/atharvaajitagwekar/Desktop/LPR/"+file) is 'png':

      image_p = cv2.imread("/Users/atharvaajitagwekar/Desktop/LPR/"+file)
      #image_p = cv2.bitwise_not(image_p)
      image_p = cv2.resize(image_p, (image_width,image_height))

      alpha = create_model()
      alpha.load_weights("./trfinal.h5")

      arr = np.array(image_p).reshape((image_width,image_height,3))
      arr = np.expand_dims(arr, axis=0)
      pred_out = alpha.predict(arr)[0]

      best_class = ''
      best_confidence = -1
      classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

      for i in classes:
          if pred_out[i]>best_confidence:
              best_class = ret_character(i)
              best_confidence = pred_out[i]

      license_number.append(best_class)

  for i in range(0,len(license_number)):
    print(license_number[i],end='')

  print("\n")

  return license_number





