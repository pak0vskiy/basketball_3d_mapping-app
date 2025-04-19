
from ultralytics import YOLO
import torch
import base64
import json
import cv2
import numpy as np
import os

class EndpointHandler():
    COURTSEG_MODEL_NAME = "court_segmentation.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    def __init__(self, path=""):
        self.path = os.path.join(path, self.COURTSEG_MODEL_NAME)
        self.court_seg_model = YOLO(self.path).to(self.device)
    
    def __call__(self, data):

        inputs = data.pop("inputs", data)
        inputs = json.loads(inputs)
        if isinstance(inputs, list):
        # Transform str into a list of base64 images
            return self.predict_batch(inputs)
        else:
            return self.predict(inputs)

    
    def predict(self, image_bytes):
        # Convert base64 to image
        if isinstance(image_bytes, str):
            # Handle base64 input
            image_bytes = base64.b64decode(image_bytes)

        
        decoded_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Run inference
        results = self.court_seg_model(decoded_image)

        if results:
            result = results[0]
        
        processed_results = {}
        
        if result.masks:
            processed_results["masks"] = result.masks.data.cpu().numpy().tolist()
            processed_results["class_ids"] = result.boxes.cls.cpu().numpy().tolist()

        
        return json.dumps(processed_results)
    
    def predict_batch(self, image_batch):

        decoded_images = []

        for image in image_batch:
            if isinstance(image, str):
            # Handle base64 input
                image_bytes = base64.b64decode(image)
                decoded_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                decoded_images.append(decoded_image)
        
        # Run inference
        batch_results = self.court_seg_model(decoded_images)

        processed_results = []
        for result in batch_results:
            processed_result = {
                "masks": result.masks.data.cpu().numpy().tolist(),
                "class_ids": result.boxes.cls.cpu().numpy().tolist()
            }
            processed_results.append(processed_result)
        return json.dumps(processed_results)




