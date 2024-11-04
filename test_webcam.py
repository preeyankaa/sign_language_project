import cv2

cap = cv2.VideoCapture(1)  # Change to 1 or 2 if necessary

if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam is accessible.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
