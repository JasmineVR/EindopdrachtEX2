import cv2
from ultralytics import YOLO
from helpers import predict_and_detect

# Load the YOLO model
def load_model():
    model = YOLO("./data/train5/weights/best.pt")
    return model

def process_webcam(model, classes=[], conf=0.5):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, _ = predict_and_detect(model, frame, classes, conf)
        cv2.imshow('Webcam', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def vision_agent(frame, model):
    # Process the frame with the model
    result_frame, results = predict_and_detect(model, frame)
    detected_behaviors = []

    # Process results to extract behavior descriptions
    for result in results:
        # Assuming results contain class names or other information
        behavior_description = f"{result['class_name']} with confidence {result['confidence']:.2f}"
        detected_behaviors.append(behavior_description)
    
    return detected_behaviors
