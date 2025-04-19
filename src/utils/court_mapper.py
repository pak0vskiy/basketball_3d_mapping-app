from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils.find_reference_points import find_points
from sklearn.cluster import KMeans
import torch
from dotenv import load_dotenv
import os
import base64
import requests
import json
import aiohttp
import asyncio

load_dotenv()
class CourtMapper:
    def __init__(self, locally=False):
        self.locally = locally
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Player detection model
        self.player_model = YOLO('model/player_detection_model.pt').to(device)
        # Court Segmentation model
        self.courtseg_model = YOLO('model/court_segmentation.pt').to(device)
        
        # Download keys
        self.player_api_url = os.environ.get("PLAYER_API_URL")
        self.court_api_url = os.environ.get("COURTSEG_API_URL")
        self.token = os.environ.get("HUGGING_FACE_KEY")
        self.headers = {
            "Accept" : "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json" 
        }

        
        # Your predefined 2D court keypoints
        self.court_2d_keypoints = {
            0: (28, 22),      # left_top_baseline_corner
            1: (28, 50),      # left_top_3_corner
            2: (28, 148),     # left_middle_baseline-top
            3: (28, 225),     # left_middle_baseline-bottom
            4: (28, 322),     # left_bottom_3_corner
            5: (28, 350),     # left_bottom_baseline_corner
            6: (147, 148),    # left_top_freethrow
            7: (147, 225),    # left_bottom_freethrow
            8: (320, 22),     # center_upper_point
            9: (320, 350),    # center_bottom_point
            10: (612, 22),    # right_top_baseline_corner
            11: (612, 50),    # right_top_3_corner
            12: (612, 148),   # right_middle_baseline-top
            13: (612, 225),   # right_middle_baseline-bottom
            14: (612, 322),   # right_bottom_3_corner
            15: (612, 350),   # right_bottom_baseline_corner
            16: (493, 148),   # right_top_freethrow
            17: (493, 225)    # right_bottom_freethrow
        }

        self.court_leftpts = np.array([
            list(self.court_2d_keypoints[8]), #TR
            list(self.court_2d_keypoints[9]), #BR
            list(self.court_2d_keypoints[5]), #BL
            list(self.court_2d_keypoints[0])  #TL
        ])
        self.court_rightpts = np.array([
            list(self.court_2d_keypoints[9]), #TL
            list(self.court_2d_keypoints[8]), #BL
            list(self.court_2d_keypoints[10]), #BR
            list(self.court_2d_keypoints[15])  #TR
        ])
        self.court_midpts = np.array([
            list(self.court_2d_keypoints[5]),
            list(self.court_2d_keypoints[9]),
            list(self.court_2d_keypoints[8]),
            list(self.court_2d_keypoints[0])
        ])

        
        # The previous homography matrix
        self.previous_leftHomography = None
        self.previous_rightHomography = None
        self.previous_midHomography = None
        
        # Add reference colors for team classification
        self.reference_team_colors = None
        self.reference_kmeans = KMeans(n_clusters=2, random_state=42)

        # Court caching
        self.court_cache_result = None
        self.court_cache_frame_count = 0
        self.court_cache_max_age = 10  # Only reuse court segmentation every N frames

        self.player_metrics = {
            0: {  # Team 0
                'total_distance': 0,
                'heatmap': np.zeros((372, 640)),  # Court dimensions 
                'possession_time': 0,
                'court_coverage': set()
            },
            1: {  # Team 1 
                'total_distance': 0,
                'heatmap': np.zeros((372, 640)),
                'possession_time': 0,
                'court_coverage': set()
            }
        }
        self.previous_positions = {0: {}, 1: {}}  # Store previous positions for each player
        self.cell_size = 32

    def update_metrics(self, court_positions, frame_idx):
        """Update player metrics for each frame"""
        current_positions = {0: {}, 1: {}}
        ball_pos = None
        
        for pos in court_positions:
            x, y, det_label, team = pos
            
            if det_label == 0:  # Player
                current_positions[team][len(current_positions[team])] = (x, y)
                
                # Update heatmap with correct bounds checking
                y_coord = max(0, min(int(y), 371))  # Height is 372
                x_coord = max(0, min(int(x), 639))  # Width is 640
                self.player_metrics[team]['heatmap'][y_coord, x_coord] += 1
                
                # Update court coverage
                zone_x = int(x / self.cell_size)
                zone_y = int(y / self.cell_size)
                self.player_metrics[team]['court_coverage'].add((zone_x, zone_y))
                
            elif det_label == 1:  # Ball
                ball_pos = (x, y)
        
        # Calculate distances
        for team in [0, 1]:
            for player_id, curr_pos in current_positions[team].items():
                if player_id in self.previous_positions[team]:
                    prev_pos = self.previous_positions[team][player_id]
                    distance = np.sqrt(
                        (curr_pos[0] - prev_pos[0])**2 + 
                        (curr_pos[1] - prev_pos[1])**2
                    )
                    # Convert pixels to meters (approximate court width is 28m)
                    distance_meters = distance * (28 / 1280)
                    self.player_metrics[team]['total_distance'] += distance_meters
            
            # Update previous positions
            self.previous_positions[team] = current_positions[team]
        
        # Update possession based on proximity to ball
        if ball_pos:
            min_dist = float('inf')
            possession_team = None
            for team in [0, 1]:
                for player_pos in current_positions[team].values():
                    dist = np.sqrt(
                        (ball_pos[0] - player_pos[0])**2 + 
                        (ball_pos[1] - player_pos[1])**2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        possession_team = team
            
            if possession_team is not None and min_dist < 100:  # Threshold for possession
                self.player_metrics[possession_team]['possession_time'] += 1

    def get_metrics_summary(self):
        """Return summary of tracked metrics"""
        return {
            team: {
                'total_distance_km': round(metrics['total_distance'] / 1000, 2),
                'possession_percentage': round(
                    metrics['possession_time'] / max(sum(
                        t['possession_time'] for t in self.player_metrics.values()
                    ), 1) * 100, 1
                ),
                'court_coverage_percentage': round(
                    len(metrics['court_coverage']) / 
                    ((1280 / self.cell_size) * (640 / self.cell_size)) * 100, 1
                ),
                'heatmap': metrics['heatmap']
            }
            for team, metrics in self.player_metrics.items()
        }

    def query(self, payload, api_url):
            response = requests.post(api_url, headers=self.headers, json=payload)
            return response.json()
    
    def query_batch(self, payloads, api_url):
        """Send batch request to the API endpoint"""
        # For batch API call, construct the appropriate payload
        batch_payload = {"inputs": payloads, "parameters": {}}
        response = requests.post(api_url, headers=self.headers, json=batch_payload)
        return response.json()
    
    async def query_batch_async(self, payloads, api_url):
        """Send batch request to the API endpoint asynchronously"""
        batch_payload = {"inputs": payloads, "parameters": {}}
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(api_url, json=batch_payload) as response:
                raw_response = await response.text()
                #print(f"Raw API response: {raw_response}")  # Debug print
                try:
                    return json.loads(raw_response)
                except json.JSONDecodeError:
                    print(f"Failed to decode response from {api_url}")
                    return None
            
    
    
    
    async def query_async(self, payload, api_url):
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(api_url, json=payload) as response:
                return await response.json()
            
    def run_async(self, coroutine):
        """Helper method to run async code"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coroutine)
    
    def compute_homography(self, src_pts, court_part: str):
        """Compute homography from detected keypoints to 2D court"""
        # Extract keypoints and confidences
        if court_part not in {"left", "right", "mid"}:
            raise AttributeError(f"Invalid court part '{court_part}'. Expected one of: 'left', 'right', 'mid'.")
        
        #src_pts = self.extract_points_from_mask(frame)
        
        if court_part == "left":
            H, _ = cv2.findHomography(src_pts, self.court_leftpts, cv2.RANSAC, 5.0)
            if self._is_valid_homography(H):
                self.previous_leftHomography = H
            return self.previous_leftHomography if self.previous_leftHomography is not None else np.eye(3)
        if court_part == "right":
            src_pts = np.array([
                src_pts[2],
                src_pts[3],
                src_pts[0],
                src_pts[1]
            ], dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, self.court_rightpts, cv2.RANSAC, 5.0)
            if self._is_valid_homography(H):
                self.previous_rightHomography = H
            return self.previous_rightHomography if self.previous_rightHomography is not None else np.eye(3)
        if court_part == "mid":
            H, _ = cv2.findHomography(src_pts, self.court_midpts, cv2.RANSAC, 5.0)
            if self._is_valid_homography(H):
                self.previous_midHomography = H
            return self.previous_midHomography if self.previous_midHomography is not None else np.eye(3)
            
        # Fallback to previous homography or identity matrix
        return self.previous_homography if self.previous_homography is not None else np.eye(3)
    
    def _is_valid_homography(self, H):
        """Check if the homography matrix is valid"""
        # Simple validation check
        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            return False
        
        # Check for reasonable scaling
        singular_values = np.linalg.svd(H[:2, :2], compute_uv=False)
        condition_number = singular_values[0] / singular_values[1]
        
        return 0.01 < condition_number < 100
    
    async def process_batch_async(self, images_base64):
        """Make concurrent API requests for both models with batch processing"""
        tasks = [
            self.query_batch_async(images_base64, self.court_api_url),
            self.query_batch_async(images_base64, self.player_api_url)
        ]
        return await asyncio.gather(*tasks)
    
    def process_batch(self, frames):
        """Process a batch of images through both models at once"""
        # Encode all images
        images_base64 = []
        images_rgb = []
        decoded_images = []
        
        for frame_path in frames:
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_rgb.append(image)
            
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            images_base64.append(image_base64)
            
        
        if self.locally:
            for image in images_base64:
                if isinstance(image, str):
                    # Handle base64 input
                    image_bytes = base64.b64decode(image)
                    decoded_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    decoded_images.append(decoded_image)

            
            court_results, player_results = self.courtseg_model(decoded_images), self.player_model(decoded_images, conf=0.4,)

            

            court_processed_results = []
            player_processed_results = []
            for result in court_results:
                processed_result = {
                    "masks": result.masks.data.cpu().numpy().tolist(),
                    "class_ids": result.boxes.cls.cpu().numpy().tolist()
                }
                court_processed_results.append(processed_result)

            for result in player_results:
                boxes = result.boxes
                processed_result = {
                    "boxes": boxes.xyxy.cpu().numpy().tolist(),
                    "conf": boxes.conf.cpu().numpy().tolist(),
                    "cls": boxes.cls.cpu().numpy().tolist()
                }
                player_processed_results.append(processed_result)

            
            return images_rgb, json.dumps(court_processed_results), json.dumps(player_processed_results)
        
        # Convert images list to str with json.dumps
        images_base64 = json.dumps(images_base64)
        
        try:
            # Get both results at once
            court_results, player_results = self.run_async(self.process_batch_async(images_base64))
            
            # Ensure results are properly decoded
            if isinstance(court_results, str):
                court_results = json.loads(court_results)
            if isinstance(player_results, str):
                player_results = json.loads(player_results)
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Court results: {court_results}")
            print(f"Player results: {player_results}")
            raise
        
        return images_rgb, court_results, player_results
    
    async def process_image_async(self, image_base64):
        """Make concurrent API requests for both models"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [
                self.query_async({"inputs": image_base64, "parameters": {}}, self.court_api_url),
                self.query_async({"inputs": image_base64, "parameters": {}}, self.player_api_url)
            ]
            return await asyncio.gather(*tasks)

    def process_image(self, frame):
        """Process image through both models at once"""
        # Read and encode image once
        image = cv2.imread(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get both results at once
        court_result, player_result = self.run_async(self.process_image_async(image_base64))
        
        return image, court_result, player_result


    def extract_points_from_mask(self, image, result):
        """Extract reference points from the court segmentation mask"""
        # Extract segmentation masks (result.masks.xy contains polygons, result.masks.data contains binary masks)
        # image = cv2.imread(frame)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode image to base64
        # First encode as jpg/png
        # _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # # Convert to base64 string
        # image_base64 = base64.b64encode(buffer).decode('utf-8')

        # result = self.run_async(self.query_async({
        #     "inputs": image_base64,
        #     "parameters": {}
        # }, self.court_api_url)
        
        #result = self.courtseg_model.predict(frame)

        img_width = image.shape[1]
        img_height = image.shape[0]
        
        if result:
            masks = np.array(result["masks"])  # Convert to NumPy array
            class_ids = np.array(result["class_ids"])  # Get class IDs for each mask

            # Define your "basketball court" class index (adjust if necessary)
            court_class_idx = 2

            # Find masks corresponding to the basketball court
            court_mask = np.zeros_like(masks[0], dtype=np.uint8)
            for i, cls in enumerate(class_ids):
                if int(cls) == court_class_idx:
                    court_mask = np.maximum(court_mask, masks[i])  # Merge court masks if multiple

            # Convert to 255 scale for saving/viewing
            court_mask = (court_mask * 255).astype(np.uint8)

            # Find dimensions scalers between input image and court mask to account for difference in size
            x_scaler = img_width / court_mask.shape[1]
            y_scaler = img_height / court_mask.shape[0]

            reference_points = find_points(court_mask)
            #Convert reference points to original image scale
            reference_points[:, 0] = reference_points[:, 0] * x_scaler
            reference_points[:, 1] = reference_points[:, 1] * y_scaler

            return reference_points
            
        else:
            return None
        



    def detect_and_map_players_batch(self, frames, court_part):
        """Detect players and map them to the 2D court with team classification"""
        # Process image once for both models
        images, court_results, player_results = self.process_batch(frames)

        if isinstance(court_results, str):
            court_results = json.loads(court_results)
        if isinstance(player_results, str):
            player_results = json.loads(player_results)

        all_court_positions = []

        for idx, (image, court_result, player_result) in enumerate(zip(images, court_results, player_results)):
            court_positions = []
            
            # Reuse court segmentation result for multiple frames to reduce computation
            # Only process court every N frames
            if idx == 0 or self.court_cache_frame_count >= self.court_cache_max_age:
                src_pts = self.extract_points_from_mask(image, court_result)
                if src_pts is not None:
                    self.court_cache_result = src_pts
                    self.court_cache_frame_count = 0
            else:
                src_pts = self.court_cache_result
                self.court_cache_frame_count += 1
                
            if src_pts is None:
                all_court_positions.append([])
                continue
        
            homography = self.compute_homography(src_pts, court_part=court_part)
            
            player_result = json.loads(player_result) if isinstance(player_result, str) else player_result
            
            
            if player_result and len(player_result) > 0:
                result = player_result
                boxes = result["boxes"]
                detection_labels = result["cls"]

                # First pass: collect player boxes and extract jersey colors
                player_boxes = []
                player_indices = []
                colors = []
                
                # frame_rgb = cv2.imread(frame)
                # frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                
                # First, collect player boxes and extract colors
                for i, (box, label) in enumerate(zip(boxes, detection_labels)):
                    if int(label) == 0:  # Player detection
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Calculate jersey region
                        box_height = y2 - y1
                        box_width = x2 - x1
                        
                        # Define jersey region
                        jersey_top = y1 + int(0.15 * box_height)
                        jersey_bottom = y1 + int(0.55 * box_height)
                        jersey_left = x1 + int(0.25 * box_width)
                        jersey_right = x2 - int(0.25 * box_width)
                        
                        # Extract jersey color
                        jersey_roi = image[jersey_top:jersey_bottom, jersey_left:jersey_right]
                        mean_color = np.mean(jersey_roi, axis=(0,1))
                        
                        colors.append(mean_color)
                        player_boxes.append(box)
                        player_indices.append(i)
                
                # Perform clustering if we have enough players
                team_labels = None
                if len(colors) > 1:
                    colors = np.array(colors)  # Ensure colors is numpy array
                    if self.reference_team_colors is None:
                        # First frame: establish reference colors
                        team_labels = self.reference_kmeans.fit_predict(colors)
                        team_labels = np.array(team_labels)
                        
                        # Use boolean indexing properly
                        mask_team_0 = (team_labels == 0)
                        mask_team_1 = (team_labels == 1)
                        
                        self.reference_team_colors = {
                            0: np.mean(colors[mask_team_0], axis=0) if np.any(mask_team_0) else np.zeros(3),
                            1: np.mean(colors[mask_team_1], axis=0) if np.any(mask_team_1) else np.zeros(3)
                        }
                    else:
                        # Subsequent frames: classify based on reference colors
                        team_labels = np.zeros(len(colors), dtype=int)
                        for i, color in enumerate(colors):
                            # Compare distance to reference colors
                            dist_0 = np.linalg.norm(color - self.reference_team_colors[0])
                            dist_1 = np.linalg.norm(color - self.reference_team_colors[1])
                            team_labels[i] = 0 if dist_0 < dist_1 else 1

                        # Update reference colors with moving average
                        alpha = 0.1  # Update rate
                        for team in [0, 1]:
                            mask_team = (team_labels == team)
                            if np.any(mask_team):
                                new_mean = np.mean(colors[mask_team], axis=0)
                                self.reference_team_colors[team] = (
                                    (1 - alpha) * self.reference_team_colors[team] + 
                                    alpha * new_mean
                                )

                # Create a mapping from original index to team label
                team_mapping = {}
                if team_labels is not None:
                    for idx, player_idx in enumerate(player_indices):
                        team_mapping[player_idx] = int(team_labels[idx])
                
                # Second pass: Map all detections to court positions
                for i, (box, det_label) in enumerate(zip(boxes, detection_labels)):
                    xyxy = box
                    foot_x = (xyxy[0] + xyxy[2]) / 2
                    foot_y = xyxy[3]
                    
                    # Apply homography
                    player_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                    court_point = cv2.perspectiveTransform(player_point, homography)
                    
                    if int(det_label) == 0:  # Player detection
                        team = team_mapping.get(i, 0)  # Get team label from mapping
                        court_positions.append((
                            int(court_point[0][0][0]), 
                            int(court_point[0][0][1]),
                            int(det_label),
                            team
                        ))
                    else:  # Ball or other objects
                        court_positions.append((
                            int(court_point[0][0][0]),
                            int(court_point[0][0][1]),
                            int(det_label),
                            -1  # No team for non-player objects
                        ))
            self.update_metrics(court_positions, idx)
            all_court_positions.append(court_positions)
        
        return all_court_positions
    
    # Keep the original method for backward compatibility
    def detect_and_map_players(self, frame, court_part):
        """Single frame processing for backward compatibility"""
        result = self.detect_and_map_players_batch([frame], court_part)
        return result[0] if result else []

