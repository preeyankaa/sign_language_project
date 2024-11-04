from flask import Flask, jsonify, render_template
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)

# Load Hand Detector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/parik/OneDrive/Documents/sign language_pps/Model1/keras_model.h5", 
                        "C:/Users/parik/OneDrive/Documents/sign language_pps/Model1/labels.txt")

labels = ["Hello", "I love you", "Peace", "Ok", "Good Luck"]

# Global variable to track whether detection is running
is_detecting = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection')
def start_detection():
    global is_detecting
    is_detecting = True
    
    # Access the webcam for real-time detection
    cap = cv2.VideoCapture(0)
    
    while is_detecting:
        success, img = cap.read()
        if not success:
            return jsonify({"prediction": "Error accessing webcam"})

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y-20:y + h + 20, x-20:x + w + 20]
            
            aspectRatio = h / w
            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                imgWhite[:, math.ceil((300 - wCal) / 2): wCal + math.ceil((300 - wCal) / 2)] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                imgWhite[math.ceil((300 - hCal) / 2): hCal + math.ceil((300 - hCal) / 2), :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            return jsonify({"prediction": labels[index]})

    cap.release()
    return jsonify({"prediction": "No hand detected"})

@app.route('/stop_detection')
def stop_detection():
    global is_detecting
    is_detecting = False
    return jsonify({"message": "Detection stopped"})

if __name__ == "__main__":
    app.run(debug=True)
