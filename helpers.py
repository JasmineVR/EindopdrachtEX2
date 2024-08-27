import cv2
import random

def draw_boxes(image, boxes, confidences, class_ids, classes, color_map):
    """
    Draw bounding boxes on the image.

    Parameters:
    - image: The image on which to draw.
    - boxes: List of bounding boxes, each represented as [x, y, width, height].
    - confidences: List of confidences for each bounding box.
    - class_ids: List of class IDs for each bounding box.
    - classes: List of class names corresponding to class IDs.
    """
    font_scale = 3  # Increased font scale
    thickness = 3  # Increased thickness for the text
    font_color = (255, 255, 255)  # Changed font color to black

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = color_map[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), color, -1)
        cv2.putText(image, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

def predict(chosen_model, img, classes=[], conf=0.5):
    """
    chosen_model: The trained model to use for prediction
    img: The image to make a prediction on
    classes: (Optional) A list of class names to filter predictions to
    conf: (Optional) The minimum confidence threshold for a prediction to be considered

    The conf argument is used to filter out predictions with a confidence score lower than the specified threshold. This is useful for removing false positives.

    The function returns a list of prediction results, where each result contains the following information:

    name: The name of the predicted class
    conf: The confidence score of the prediction
    box: The bounding box of the predicted object
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def generate_color_map(num_classes):
    """
    Generate a color map for the given number of classes.

    Parameters:
    - num_classes: The number of classes.

    Returns:
    - A dictionary mapping class IDs to colors.
    """
    random.seed(42)  # For reproducibility
    color_map = {}
    for i in range(num_classes):
        color_map[i] = [random.randint(0, 255) for _ in range(3)]
    return color_map

# predict_and_detect function

def predict_and_detect(model, frame, classes=[], conf=0.5):
    # Run inference
    results = model(frame)

    detections = []

    # Process results
    for result in results:
        # Extracting bounding boxes, class labels, and confidences
        for box in result.boxes:
            if box.conf >= conf:
                # Convert tensor to integer for class index
                class_id = int(box.cls.item())
                detection = {
                    'class_name': model.names[class_id],
                    'confidence': box.conf.item(),
                    'box': box.xyxy # Get confidence
                }
                detections.append(detection)
    
    # Optionally draw bounding boxes on the frame
    result_frame = frame

    return result_frame, detections
