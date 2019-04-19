import numpy 
import cv2
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.models import Sequential as seq
from keras.layers import Conv2D
from keras.layers import MaxPooling2D as MaxP2D
from keras.layers import Dense, Flatten,Dropout
import os
import sys
import imghdr

image_width, image_height = 128,128

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

def ret_ch(i):
    chars=['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return chars[i]

def ret_pred(file):

  image_p = cv2.imread(file)
  image_p = cv2.resize(image_p, (image_width,image_height))

  alpha = create_model()
  alpha.load_weights("./trfinal.h5")

  arr = numpy.array(image_p).reshape((image_width,image_height,3))
  arr = numpy.expand_dims(arr, axis=0)
  pred_out = alpha.predict(arr)[0]

  best_class = ''
  best_confidence = -1

  classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

  for i in classes:
    if pred_out[i]>best_confidence:
      best_class = ret_ch(i)
      best_confidence = pred_out[i]

  return best_class

