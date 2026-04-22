from ultralytics import YOLO
import cv2

# ----- CHANGE THESE -----
MODEL_PATH = "C:\\NEWDRIVE\\Model_train\\best.pt"
IMAGE_PATH = "C:\\NEWDRIVE\\Model_train\\Yolo_try\\test_images\\img_258.jpg"
# ------------------------

model = YOLO(MODEL_PATH)

# Run prediction
results = model.predict(source=IMAGE_PATH, conf=0.25)

# Get first result
r = results[0]

# Draw segmentation automatically
img = r.plot()

# Show image
cv2.imshow("Segmentation Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("segmentation_result.jpg", img)

print("Result saved as segmentation_result.jpg")