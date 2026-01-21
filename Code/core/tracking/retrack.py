import torch
import numpy as np
from boxmot import StrongSort
from pathlib import Path
import config

class Tracker:
    def __init__(self):
        print("🔗 Tracking: Initializing StrongSORT++...")
        
        # Load StrongSORT with ReID
        self.tracker = StrongSort(
            model_weights=config.REID_WEIGHTS,
            device=torch.device(config.DEVICE),
            half=config.FP16,
            max_age=70 # Persistence (Hold ID for 70 frames if occluded)
        )

    def update(self, p_boxes, c_boxes, frame):
        """
        Merges detections and updates the tracker.
        Returns: [x1, y1, x2, y2, id, conf, class, ...]
        """
        # Merge lists (if they exist)
        dets = []
        if len(p_boxes) > 0: dets.append(p_boxes)
        if len(c_boxes) > 0: dets.append(c_boxes)
        
        if not dets:
            return np.empty((0, 8)) # Return empty array if nothing found
            
        all_dets = np.vstack(dets)

        # Update Tracker
        tracks = self.tracker.update(all_dets, frame)
        
        # tracks format: [x1, y1, x2, y2, id, conf, class, ...]
        return tracks