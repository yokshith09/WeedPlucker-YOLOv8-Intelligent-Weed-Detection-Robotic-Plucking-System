from ultralytics import YOLO
import cv2

# 🔴 CHANGE THESE PATHS
MODEL_PATH = r"C:\NEWDRIVE\Model_train\Yolo_try\runs\segment\weed_detect_v3\weights\best.pt"
IMAGE_PATH = r"C:\NEWDRIVE\Model_train\Yolo_try\test_images\img_259.jpg"

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model(IMAGE_PATH)

# Get result
result = results[0]

# Draw predictions (boxes + masks automatically)
annotated = result.plot()

# Show image
cv2.imshow("Prediction", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("output1.jpg", annotated)

print("✅ Done — result saved as output1.jpg")