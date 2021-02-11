import os
import cv2
import pandas as pd

def readMyData(myListLen, pathToData):
    count = 0
    images = []
    classNo = []
    print("Total Classes Detected: ", myListLen)
    print("Importing classes...")
    for x in range(0, myListLen):
        myPicList = os.listdir("{0}/{1}".format(pathToData, str(count)))
        for y in myPicList:
            currImg = cv2.imread("{0}/{1}/{2}".format(pathToData, str(count), y))
            images.append(currImg)
            classNo.append(count)
        count += 1
    return images, classNo

def readCsv(file):
    return pd.read_csv(file)
