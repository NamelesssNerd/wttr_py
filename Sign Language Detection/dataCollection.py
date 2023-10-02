import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/A"
counter = 0

while True:
    success, img = cap.read()
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

                else:
                    k = imgSize / w1
                    hCal = math.ceil(k * h1)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

            except BaseException as e:
                print(f"Error resizing image: {e}")
            
            cv2.imshow("Joint Image", imgWhite)

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

                else:
                    k = imgSize / w1
                    hCal = math.ceil(k * h1)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

            except BaseException as e:
                print(f"Error resizing image: {e}")

            cv2.imshow("Single Image", imgWhite)

    cv2.imshow("Original Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
