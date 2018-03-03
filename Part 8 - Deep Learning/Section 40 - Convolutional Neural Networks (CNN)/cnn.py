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