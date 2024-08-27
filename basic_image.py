import cv2
from ultralytics import YOLO
from helpers import predict_and_detect

# model = YOLO("yolov9c.pt")
model = YOLO("./data/train5/weights/best.pt")

# read the image
image = cv2.imread("./samples/images/sample1.jpeg")
#image = cv2.imread("./samples/images/sample2.jpg")

# predict and detect
result_img, _ = predict_and_detect(model, image, classes=[], conf=0.15)

# display the image
cv2.imshow("Image", result_img)

# save the image
cv2.imwrite("./samples/images/sample1-analysed.jpeg", result_img)
#cv2.imwrite("./samples/images/sample2-analysed.jpeg", result_img)

# wait for a key press
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()
