import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model\keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

folder = "Data/Hello"
counter = 0

labels=["A","B", "C", "D", "E", "F", "G","H","I", "L", "O", "P", "R", "U","W", "Y","2", "4", "5", "Heart", "House", "Namaste"]

while True:
    success, img = cap.read()
    imgOutput=img.copy()
    hands, img = detector.findHands(img)

    if hands:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        leftHand = hands[0]
        x1, y1, w1, h1 = leftHand['bbox']
        if len(hands) > 1:
            rightHand = hands[1]
            x2, y2, w2, h2 = rightHand['bbox']

            if (x2 + w2 +offset> x1-offset) and (x2 - offset < x1+w1+offset):
                imgCrop = img[min(y2 - offset, y1 - offset):max(y2 + h2 + offset, y1 + h1 + offset),
                              min(x1 - offset, x2 - offset):max(x1 + w1 + offset, x2 + w2 + offset)]
            else:
                imgCrop = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]

            try:
                imgCropShape = imgCrop.shape
                h1 = imgCropShape[0]
                w1 = imgCropShape[1]

                aspectRatio = h1 / w1
                if aspectRatio > 1:
                    k = imgSize / h1
                    wCal = math.ceil(k * w1)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)

                else:
                    k = imgSize / w1
                    hCal = math.ceil(k * h1)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)

                # Calculate the bounding box that contains both hands
                x_combined = min(x1 - offset, x2 - offset)
                y_combined = min(y1 - offset, y2 - offset)
                w_combined = max(x1 + w1 + offset, x2 + w2 + offset) - x_combined
                h_combined = max(y1 + h1 + offset, y2 + h2 + offset) - y_combined

                # Draw the bounding box and display text
                cv2.rectangle(imgOutput, (x_combined, y_combined - 50),
                            (x_combined + w_combined, y_combined + h_combined), (255, 0, 255))
                cv2.putText(imgOutput, labels[index], (x_combined, y_combined - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                
            except BaseException as e:
                print(f"Error resizing image: {e}")
            
            # cv2.imshow("Joint Image", imgWhite)

        else:
            imgCrop = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]

            try:
                imgCropShape = imgCrop.shape
                h1 = imgCropShape[0]
                w1 = imgCropShape[1]

                aspectRatio = h1 / w1
                if aspectRatio > 1:
                    k = imgSize / h1
                    wCal = math.ceil(k * w1)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
                    
                else:
                    k = imgSize / w1
                    hCal = math.ceil(k * h1)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
                
                cv2.rectangle(imgOutput, (x1 - offset, y1 - offset-50),
                      (x1 - offset+90, y1 - offset-50+50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x1, y1 -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x1-offset, y1-offset),
                      (x1 + w1+offset, y1 + h1+offset), (255, 0, 255), 4)
            except BaseException as e:
                print(f"Error resizing image: {e}")

    cv2.imshow("Original Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)