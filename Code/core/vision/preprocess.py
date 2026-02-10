import cv2
import numpy as np
import config
from pathlib import Path

class FramePreprocessor:
    def __init__(self):
        print("[INFO] Preprocessing: Using Hybrid CPU Pipeline (Optimized)...")
        
        # 1. Setup CLAHE
        if config.ENABLE_CLAHE:
            self.clahe = cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT, 
                tileGridSize=config.CLAHE_GRID_SIZE
            )
        
        # 2. Load Mask (CPU)
        self.mask = None
        if config.ROI_MASK_PATH and Path(config.ROI_MASK_PATH).exists():
            self.mask = cv2.imread(str(config.ROI_MASK_PATH), 0)
            if self.mask is not None:
                print(f"[INFO] ROI Mask Loaded: {config.ROI_MASK_PATH}")

    def process(self, frame):
        """
        Input: Resized BGR Frame (CPU)
        Output: Processed BGR Frame (CPU) ready for YOLO
        """
        processed = frame.copy()

        # 1. Median Blur
        if config.ENABLE_MEDIAN_BLUR:
            processed = cv2.medianBlur(processed, config.BLUR_KERNEL_SIZE)

        # 2. CLAHE (Contrast)
        if config.ENABLE_CLAHE:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Apply Mask
        if self.mask is not None:
            # Resize mask to match current frame dimensions
            if self.mask.shape != processed.shape[:2]:
                self.mask = cv2.resize(self.mask, (processed.shape[1], processed.shape[0]))
            processed = cv2.bitwise_and(processed, processed, mask=self.mask)

        return processed