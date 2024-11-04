import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/parik/OneDrive/Documents/sign language_pps/Model2/keras_model.h5", 
                        "C:/Users/parik/OneDrive/Documents/sign language_pps/Model2/labels.txt")
offset = 20
imgSize = 300

labels = ["Call Me","Crime","Good Luck","Greet Fight","Greet Support","Hate","Hello","Hurts a lot","I Love You","mini heart","No","Ok","Peace","Pinch","Pointing left","Pointing up","Receive","Rock","Thank You","Yes"]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read image from camera.")
        continue  # Skip the rest of the loop if there's an error

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the crop coordinates are within the image boundaries
        y_start = max(0, y - offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(img.shape[1], x + w + offset)

        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size == 0:
            print("Error: imgCrop is empty.")
            continue  # Skip to the next iteration

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgWhite[:, math.ceil((imgSize - wCal) / 2): wCal + math.ceil((imgSize - wCal) / 2)] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgWhite[math.ceil((imgSize - hCal) / 2): hCal + math.ceil((imgSize - hCal) / 2), :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)

    # Allow quitting the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
