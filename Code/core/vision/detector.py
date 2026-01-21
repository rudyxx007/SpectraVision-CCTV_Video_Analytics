from ultralytics import YOLO
import torch
import config 

class DualDetector:
    def __init__(self):
        print(f"👁️ Vision: Loading YOLO26-Pose & YOLO26-Nano on {config.DEVICE}...")
        
        # Model 1: People & Skeletons
        self.pose_model = YOLO(config.POSE_WEIGHTS)
        
        # Model 2: Furniture (Chairs)
        # We use standard YOLO26n because it's trained on COCO (Class 56 = Chair)
        self.chair_model = YOLO(config.CHAIR_WEIGHTS)
        
        print("✅ Vision: Models Loaded.")

    def detect(self, frame):
        """
        Returns:
        1. person_boxes: [x1, y1, x2, y2, conf, cls_id=0]
        2. chair_boxes:  [x1, y1, x2, y2, conf, cls_id=56]
        3. keypoints:    [N, 17, 3] (Skeleton data)
        """
        # --- A. People (Pose) ---
        p_results = self.pose_model(frame, verbose=False, device=config.DEVICE, conf=config.CONF_THRESH)
        person_boxes = p_results[0].boxes.data.cpu().numpy() 
        keypoints = p_results[0].keypoints.data.cpu().numpy() 
        
        # --- B. Chairs (Object) ---
        # Filter for Class 56 (Chair) immediately
        c_results = self.chair_model(frame, verbose=False, device=config.DEVICE, classes=[56], conf=0.3)
        chair_boxes = c_results[0].boxes.data.cpu().numpy()

        return person_boxes, chair_boxes, keypoints