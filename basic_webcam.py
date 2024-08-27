import cv2
import random
from ultralytics import YOLO
from helpers import predict_and_detect

#model
model = YOLO("./data/train5/weights/best.pt")

def process_webcam(model, classes=[], conf=0.5):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        result_frame, _ = predict_and_detect(model, frame, classes, conf)

        # Display the processed frame
        cv2.imshow('Webcam', result_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_webcam(model, classes=[], conf=0.5)
