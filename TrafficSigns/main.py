from typing import List

import numpy as np
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
import preprocesingImages as ppi
import util as ut
import Constants

# importovanje slika iz foldera myData
count = 0
images = []
classNo = []
myList = os.listdir(Constants.PATH_TO_DATA)
myListLen = len(myList)

images, classNo = ut.readMyData(myListLen, Constants.PATH_TO_DATA)
images = np.array(images)
classNo = np.array(classNo)

# Deljenje podataka na train/test/validation (X_train = niz slika za treniranje, y_train = odgovarajuci ID klasa)
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=Constants.TEST_RATIO)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=Constants.VALIDATION_RATIO)

## provera da li se broj slika poklapa sa brojem labela za svaki data-set
print("Data Shapes")
print("Train", end="");
print(X_train.shape, y_train.shape)
print("Validation", end="");
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)

# ucitavanje labela
data = ut.readCsv(Constants.LABELS_FILE_NAME)
print("data shape ", data.shape, type(data))

# Iteriranje i preprocesiranje svih slika
X_train = np.array(list(map(ppi.preprocessing, X_train)))
X_validation = np.array(list(map(ppi.preprocessing, X_validation)))
X_test = np.array(list(map(ppi.preprocessing, X_test)))

# dodavanje depth-a 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)