import numpy as np
import cv2
import Constants
import pickle
import TrafficSigns.helpers.preprocesingImages as ppi
import TrafficSigns.helpers.util as ut

# postavljanje parametara video kamere
cap = cv2.VideoCapture(0)
cap.set(3, Constants.FRAME_WIDTH)
cap.set(4, Constants.FRAME_HEIGHT)
cap.set(10, Constants.BRIGHTNESS)

# import-ovanje istreniranog modela
pickle_in = open("model_trained_1.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)

while True:
    # ucitavanje slike
    success, imgOrignal = cap.read()

    # procesiranje slike
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = ppi.preprocessing(img)
    cv2.imshow("Procesirana slika", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "Klasa: ", (20, 35), Constants.FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "Verovatnoca: ", (20, 75), Constants.FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # predikcija slike
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > Constants.THRESHOLD:
        cv2.putText(imgOrignal, str(classIndex) + " " + str(ut.getClassName(classIndex)), (120, 35), Constants.FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), Constants.FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Rezultat", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break