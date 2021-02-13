from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from TrafficSigns import Constants
import pickle
import cv2

def makingModel(myListLen):
    model = Sequential()
    model.add((Conv2D(Constants.NO_OF_FILTERS, Constants.SIZE_OF_FILTER,
                      input_shape=(Constants.IMAGE_DIMENSIONS[0],
                                   Constants.IMAGE_DIMENSIONS[1], 1), activation='relu')))
    model.add((Conv2D(Constants.NO_OF_FILTERS, Constants.SIZE_OF_FILTER, activation='relu')))
    model.add(MaxPooling2D(pool_size=Constants.SIZE_OF_POOL))  # ne utice na DEPTH/broj filtera

    model.add((Conv2D(Constants.NO_OF_FILTERS // 2, Constants.SIZE_OF_FILTER2, activation='relu')))
    model.add((Conv2D(Constants.NO_OF_FILTERS // 2, Constants.SIZE_OF_FILTER2, activation='relu')))
    model.add(MaxPooling2D(pool_size=Constants.SIZE_OF_POOL))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(Constants.NO_OF_NODES, activation='relu'))
    model.add(Dropout(0.5))  # postavljanje random ulaza na 0, sa verovatnocom 0.5
    model.add(Dense(myListLen, activation='softmax'))  # izlazni sloj

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # kompajliranje modela
    return model

def writeModel(model):
    pickle_out = open("model_trained" + str(Constants.EPOCHS_NUMBER) + ".p", "wb")  # write byte
    pickle.dump(model, pickle_out)
    pickle_out.close()
    cv2.waitKey(0)