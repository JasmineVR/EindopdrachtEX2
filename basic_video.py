import cv2
from ultralytics import YOLO
from helpers import predict_and_detect

# Load YOLO model
model = YOLO("yolov9c.pt")

# Open video file
video_capture = cv2.VideoCapture("./samples/video/cars.mp4")

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
        # End of the video
        break  # Break the loop if there are no more frames

    # Predict and detect objects in the frame
    result_img, _ = predict_and_detect(model, frame, classes=[], conf=0.5)

    # Display the resulting frame
    cv2.imshow("Video", result_img)

    # Check for the 'q' key pressed or if the window is closed to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' key or ESC key
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
