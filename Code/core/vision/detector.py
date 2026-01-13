import torch
import cv2
import numpy as np
from sam3 import sam3_model_registry, Sam3Predictor

# If you have InternVL weights, uncomment imports below. 
# For now, let's just get SAM 3 working to see progress.
# from transformers import AutoModel, AutoTokenizer

class VisionSystem:
    def __init__(self, sam_checkpoint):
        print(f"🚀 Initializing Vision on {torch.cuda.get_device_name(0)}...")
        
        # 1. Initialize SAM 3
        # BFloat16 is the "Secret Weapon" for RTX 50-series speed
        self.sam = sam3_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda", dtype=torch.bfloat16)
        
        # Compile it for Blackwell (RTX 50) optimization
        # This replaces the need for 'transformer_engine'
        print("⚡ Compiling Model for RTX 5050...")
        self.sam = torch.compile(self.sam, mode="reduce-overhead")
        
        self.predictor = Sam3Predictor(self.sam)

    def process_frame(self, frame):
        """
        Input: BGR Frame from OpenCV
        Output: Masks
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use Mixed Precision (AMP) for speed
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            self.predictor.set_image(rgb_frame)
            
            # Simple prompt: Find People and Chairs
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=True
            )
            
        return masks

print("✅ Detector Logic Loaded.")