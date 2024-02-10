import os
from keras.models import load_model
import cv2
import numpy as np
import yolov8_detection
image='image.jpg'
save=yolov8_detection.detect_save_image(image)


model = load_model(r"D:\kk\best_model.h5")

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    return img

for i in os.listdir(r'output_faces'):
    image_path = f"output_faces/{i}"
    print(image_path)

    input_image = load_and_preprocess_image(image_path)

    predictions = model.predict(np.array([input_image]))

    class_labels = person_name_list 
    predicted_class = class_labels[np.argmax(predictions)]
    print(np.argmax(predictions))
    print(predicted_class)
    # print(f"The model predicts that the image is a {predicted_class}")
