import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import config

class SegmentationEngine:
    def __init__(self):
        print("[INFO] Segmentor: Initializing SAM (Precision Masking)...")
        try:
            # We use vit_b (Base) for speed/accuracy balance
            self.sam = sam_model_registry["vit_b"](checkpoint=str(config.SAM_WEIGHTS))
            self.sam.to(device=config.DEVICE)
            self.predictor = SamPredictor(self.sam)
            print("[SUCCESS] SAM Initialized.")
        except Exception as e:
            print(f"[ERROR] SAM Init Failed (Check weights path): {e}")
            self.predictor = None

    def get_mask_iou(self, frame, boxA, boxB):
        """
        Calculates Pixel-wise IoU between two objects using SAM.
        Input: Frame, BoxA (Person), BoxB (Chair)
        Output: Mask IoU Score (0.0 to 1.0)
        """
        if self.predictor is None: return 0.0

        # Set image once
        self.predictor.set_image(frame)

        # Predict Mask A (Person)
        maskA, _, _ = self.predictor.predict(
            box=np.array(boxA), multimask_output=False
        )
        
        # Predict Mask B (Chair)
        maskB, _, _ = self.predictor.predict(
            box=np.array(boxB), multimask_output=False
        )

        # Calculate Intersection/Union on pixels
        intersection = np.logical_and(maskA[0], maskB[0]).sum()
        union = np.logical_or(maskA[0], maskB[0]).sum()

        if union == 0: return 0.0
        return intersection / union