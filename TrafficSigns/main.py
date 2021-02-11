import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import ImageDataGenerator
from TrafficSigns.helpers import preprocesingImages as ppi, modelUtil as mu, util as ut
from TrafficSigns import Constants

# Importovanje dataSet-a
myList = os.listdir(Constants.PATH_TO_DATA)
myListLen = len(myList)

images, classNo = ut.readMyData(myListLen, Constants.PATH_TO_DATA)
images = np.array(images)
classNo = np.array(classNo)

# Deljenje podataka na train/test/validation (X_train = niz slika za treniranje, y_train = odgovarajuci ID klasa)
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=Constants.TEST_RATIO)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=Constants.VALIDATION_RATIO)

## Provera da li se broj slika poklapa sa brojem labela za svaki data-set
print("Data Shapes")
print("Train", end="");
print(X_train.shape, y_train.shape)
print("Validation", end="");
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)

# Ucitavanje labela
data = ut.readCsv(Constants.LABELS_FILE_NAME)
print("data shape ", data.shape, type(data))

# Iteriranje i preprocesiranje svih slika
X_train = np.array(list(map(ppi.preprocessing, X_train)))
X_validation = np.array(list(map(ppi.preprocessing, X_validation)))
X_test = np.array(list(map(ppi.preprocessing, X_test)))

# Dodavanje depth-a 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Augmentacija slika, s ciljem vece genericnosti  (0.1 = 10%)
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 znaci da ide od 0.8 do 1.2
                             shear_range=0.1,  # procenat smicanja
                             rotation_range=10)  # rotacija (u stepenima)
dataGen.fit(X_train)

# Zahtevanje od DataGenerator-a da generise slike
# BATCH SIZE = broj slika koje se kreiraju, svaki put kad se pozove
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

y_train = to_categorical(y_train, myListLen)
y_validation = to_categorical(y_validation, myListLen)
y_test = to_categorical(y_test, myListLen)

# Pravljenje CNN modela
model = mu.makingModel(myListLen)

# Treniranje modela
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=Constants.BATCH_SIZE),
                              steps_per_epoch=Constants.STEPS_PER_EPOCH, epochs=Constants.EPOCHS_NUMBER,
                              validation_data=(X_validation, y_validation), shuffle=1)
# Evaluacija
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: ', score[0])
print('Test Accuracy: ', score[1])

# Upisivanje modela kao pickle objekta
mu.writeModel(model)