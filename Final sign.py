import cv2
import numpy as np
import math
#import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cam = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:\\Users\\bhuva\\Desktop\\MIni pro\\Final model\\keras_model.h5", "C:\\Users\\bhuva\\Desktop\\MIni pro\\Final model\\labels.txt")
offset = 20
imgSize = 300

#folder = "Computer Vision\\Data"
counter=0
labels = ["A" ,"E","H","O","R","Hello", "Yes", "No", "I love you", "Help"]
#cam.release()
#cv2.destroyAllWindows()
try:
    while True:
        success, img = cam.read()
        if(success):
            
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
        
                imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
                imgCropShape = imgCrop.shape
        
                aspectRatio = h/w
                if aspectRatio > 1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap:wCal+wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite,draw=False)
                cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+50, y-offset-50+50), (255,0,255),cv2.FILLED)#x-offset+250,
                cv2.putText(imgOutput, labels[index], (x,y-26), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2) #1.7
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x +w+offset, y + h + offset), (255, 0, 255), 4)
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageCrop", imgWhite)
            cv2.imshow("Image", imgOutput)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cam.release()
    cv2.destroyAllWindows()
except:
    print("image out of bound")
    cam.release()
    cv2.destroyAllWindows()
    
