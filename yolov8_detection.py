#!pip install ultralytics==8.0.0
import cv2
import os
from ultralytics import YOLO
from PIL import Image
yolo_model = YOLO(r'D:\kk\face-detection-yolov8\yolov8n-face.pt')
def detect_save_image(image):
    output_directory = r'output_faces'
    os.makedirs(output_directory, exist_ok=True)
    image_path = image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(source=image_path)[0]
    if results.boxes.data.tolist():
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.3: 
                face_region = image[int(y1):int(y2), int(x1):int(x2)]
                face_output_path = os.path.join(output_directory, f"face_{score:.2f}.jpg")
                cv2.imwrite(face_output_path, cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))

                
