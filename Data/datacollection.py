import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Hand detector with max 2 hands
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
counter = 0
folder = "C:/Users/parik/OneDrive/Documents/sign language_pps/Data/Hate"

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from webcam.")
        break
    
    # Detect hands
    hands, img = detector.findHands(img)

    # If at least one hand is detected
    if hands:
        hand = hands[0]  # Only consider the first hand for now
        x, y, w, h = hand['bbox']  # Get the bounding box

        # Create a white image of size 300x300
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the detected hand with a margin (offset)
        try:
            imgCrop = img[y-offset : y + h + offset, x-offset : x + w + offset]

            # Check if the cropped image is valid
            if imgCrop.size == 0:
                print("Error: imgCrop is empty or out of bounds.")
                continue  # Skip this iteration if the crop is not valid

            aspectratio = h / w

            # Resize and paste onto imgWhite based on the aspect ratio
            if aspectratio > 1:  # Height is greater than width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize  # Place resized image in the center

            else:  # Width is greater than height
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize  # Place resized image in the center

            # Show cropped and processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print(f"Error cropping or resizing image: {e}")
            continue

    # Show the original image with hands
    cv2.imshow("Image", img)

    # Press 's' to save the image or 'q' to quit
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        save_path = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(save_path, imgWhite)
        print(f"Image saved: {save_path}")
    elif key == ord("q"):  # Press 'q' to quit
        print("Quitting...")
        break  # Break out of the loop if 'q' is pressed

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
