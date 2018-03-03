"""
Download Dataset here
http://www.superdatascience.com/wp-content/uploads/2017/03/Convolutional-Neural-Networks.zip
Size: 222MB
"""
 
import pandas as pd
import numpy as np
import matplotlib as plt

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#First Convolutional Layer
#Adding Layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Second Convolutional Layer - Optional
"""
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
"""
#Flattening into a single feature vector
classifier.add(Flatten())