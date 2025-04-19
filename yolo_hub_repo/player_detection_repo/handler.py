
from ultralytics import YOLO
import torch
import base64
import json
import cv2
import numpy as np
import os

class EndpointHandler():
    PLAYERDET_MODEL_NAME = "player_detection_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    def __init__(self, path=""):
        self.path = os.path.join(path, self.PLAYERDET_MODEL_NAME)
        self.player_det_model = YOLO(self.path).to(self.device)
    
    def __call__(self, data):

        inputs = data.pop("inputs", data)
        inputs = json.loads(inputs)
        if isinstance(inputs, list):
        # Transform str into a list of base64 images
            return self.batch_predict(inputs)
        else:
            return self.predict(inputs)

    
    def predict(self, image_bytes):
        # Convert base64 to image
        if isinstance(image_bytes, str):
            # Handle base64 input
            image_bytes = base64.b64decode(image_bytes)

        
        decoded_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Run inference
        results = self.player_det_model(decoded_image, conf=0.4)
        
        processed_results = {}
        for result in results:
            boxes = result.boxes
            processed_results["boxes"] = boxes.xyxy.cpu().numpy().tolist()
            processed_results['conf'] = boxes.conf.cpu().numpy().tolist()
            processed_results['cls'] = boxes.cls.cpu().numpy().tolist()

        
        return json.dumps(processed_results)
    
    def batch_predict(self, image_batch):
        # Convert base64 to image
        decoded_images = []
        for image in image_batch:
            if isinstance(image, str):
                # Handle base64 input
                image_bytes = base64.b64decode(image)
                decoded_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                decoded_images.append(decoded_image)
            #print(len(decoded_images))
            # Run inference
        batch_results = self.player_det_model(decoded_images, conf=0.4)
        
        processed_results = []
        for result in batch_results:
            boxes = result.boxes
            processed_result = {
                "boxes": boxes.xyxy.cpu().numpy().tolist(),
                "conf": boxes.conf.cpu().numpy().tolist(),
                "cls": boxes.cls.cpu().numpy().tolist()
            }
            processed_results.append(processed_result)
        return json.dumps(processed_results)




