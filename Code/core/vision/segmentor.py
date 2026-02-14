import torch
import numpy as np
from ultralytics import SAM
import config

class SegmentationEngine:
    def __init__(self):
        print(f"[INFO] Segmentor: Initializing SAM 3 from {config.SAM_WEIGHTS}...")
        try:
            # Ultralytics natively supports SAM 3 .pt weights
            self.model = SAM(str(config.SAM_WEIGHTS))
            print("[SUCCESS] SAM 3 Initialized.")
        except Exception as e:
            print(f"[ERROR] SAM 3 Init Failed (Check weights): {e}")
            self.model = None

    def get_mask_iou(self, frame, boxA, boxB):
        if self.model is None: return 0.0
        
        try:
            # SAM 3 natively accepts bounding box prompts
            results = self.model(frame, bboxes=[boxA.tolist(), boxB.tolist()], verbose=False)
            
            if not results or results[0].masks is None: return 0.0
            
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) < 2: return 0.0
            
            mA, mB = masks[0].astype(bool), masks[1].astype(bool)
            
            # Intersection over Union for pixels
            intersection = np.logical_and(mA, mB).sum()
            union = np.logical_or(mA, mB).sum()
            
            return intersection / (union + 1e-6)
        except Exception as e:
            print(f"[WARNING] SAM 3 Inference Error: {e}")
            return 0.0