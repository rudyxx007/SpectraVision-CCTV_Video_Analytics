import cv2
import numpy as np
import config
from pathlib import Path

class FramePreprocessor:
    def __init__(self):
        print("[INFO] Initializing Image Preprocessor...")
        
        # 1. Setup CLAHE (Contrast Enhancement)
        if config.ENABLE_CLAHE:
            self.clahe = cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT, 
                tileGridSize=config.CLAHE_GRID_SIZE
            )
            print(f"[INFO] CLAHE Enabled (Clip: {config.CLAHE_CLIP_LIMIT})")
        
        # 2. Load Region of Interest (ROI) Mask
        self.mask = None
        if config.ROI_MASK_PATH and Path(config.ROI_MASK_PATH).exists():
            # Load as Grayscale (0)
            self.mask = cv2.imread(str(config.ROI_MASK_PATH), 0)
            
            # Verify mask exists
            if self.mask is not None:
                print(f"[INFO] ROI Mask Loaded: {config.ROI_MASK_PATH}")
            else:
                print(f"[WARNING] Mask file found but could not be loaded.")
        else:
            print("[INFO] No ROI Mask found. Processing full frame.")

    def process(self, frame):
        """
        Applies the configured filters to the raw video frame.
        Input: Raw BGR Frame
        Output: Processed BGR Frame ready for YOLO
        """
        # Work on a copy to avoid modifying the original
        processed = frame.copy()

        # --- Step 1: Noise Reduction (Median Blur) ---
        if config.ENABLE_MEDIAN_BLUR:
            # Reduces salt-and-pepper noise common in CCTV
            processed = cv2.medianBlur(processed, config.BLUR_KERNEL_SIZE)

        # --- Step 2: Contrast Enhancement (CLAHE) ---
        # CLAHE works on the Lightness channel (L) of LAB color space.
        if config.ENABLE_CLAHE:
            # Convert BGR -> LAB
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            l2 = self.clahe.apply(l)
            
            # Merge and convert back to BGR
            lab = cv2.merge((l2, a, b))
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # --- Step 3: Apply ROI Mask ---
        # Everything purely black (0) in the mask becomes black in the frame
        if self.mask is not None:
            # Resize mask to match frame if dimensions differ (Safety Check)
            if self.mask.shape != processed.shape[:2]:
                self.mask = cv2.resize(self.mask, (processed.shape[1], processed.shape[0]))
            
            # Bitwise AND keeps the region where mask is white
            processed = cv2.bitwise_and(processed, processed, mask=self.mask)

        return processed