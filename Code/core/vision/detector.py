from ultralytics import YOLO
import torch
import numpy as np
import config 

class DualDetector:
    def __init__(self):
        print("[INFO] Vision: Loading Perception Stack...")
        
        # Model 1: Pose Estimation (The "Skeleton" Expert)
        # Finds People (Class 0) and their Keypoints
        try:
            self.pose_model = YOLO(config.POSE_MODEL_WEIGHTS)
            print(f"[INFO] Pose Model Loaded: {config.POSE_MODEL_WEIGHTS}")
        except Exception as e:
            print(f"[ERROR] Failed to load Pose Model: {e}")
            raise
        
        # Model 2: Furniture Model (The "Object" Expert)
        # We use YOLO26-Nano to find 'Chair' (Class 56).
        try:
            self.furniture_model = YOLO(config.CHAIR_MODEL_WEIGHTS)
            print(f"[INFO] Furniture Model Loaded: {config.CHAIR_MODEL_WEIGHTS}")
        except Exception as e:
            print(f"[ERROR] Failed to load Furniture Model: {e}")
            raise
        
        print("[SUCCESS] Dual-Vision System Initialized.")

    def detect(self, frame):
        """
        Runs both models on the frame.
        Returns: 
        1. Person Boxes (from Pose Model)
        2. Chair Boxes (from Nano Model)
        3. Keypoints (from Pose Model)
        """
        
        # --- A. Run Pose Model (People) ---
        # We use the specific confidence threshold from config
        # verbose=False keeps the console clean
        pose_results = self.pose_model(frame, verbose=False, device=config.DEVICE, conf=config.CONF_THRESH)
        
        # Extract Data: [x1, y1, x2, y2, conf, class_id]
        if len(pose_results) > 0 and pose_results[0].boxes.data is not None:
            person_boxes = pose_results[0].boxes.data.cpu().numpy() 
            # Extract Keypoints: [N, 17, 3] (x, y, visibility)
            keypoints = pose_results[0].keypoints.data.cpu().numpy() 
        else:
            person_boxes = np.empty((0, 6))
            keypoints = np.empty((0, 17, 3))
        
        # --- B. Run Object Model (Chairs) ---
        # We restrict detection to Class 56 (Chair) only
        chair_results = self.furniture_model(frame, verbose=False, device=config.DEVICE, classes=[56], conf=0.3)
        
        if len(chair_results) > 0 and chair_results[0].boxes.data is not None:
            chair_boxes = chair_results[0].boxes.data.cpu().numpy()
        else:
            chair_boxes = np.empty((0, 6))

        return person_boxes, chair_boxes, keypoints