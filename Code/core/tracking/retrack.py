import torch
import numpy as np
from boxmot import StrongSort
from pathlib import Path
import config

class Tracker:
    def __init__(self):
        print("[INFO] Tracking: Initializing StrongSORT++...")
        
        # Initialize StrongSORT
        # This uses the ReID model (OSNet) to remember what people look like.
        # If 'Person 1' walks behind a pillar and comes out, this model recognizes their clothes
        # and keeps the ID as 'Person 1' instead of switching to 'Person 2'.
        self.tracker = StrongSort(
            reid_weights=config.REID_WEIGHTS,
            device=torch.device(config.DEVICE),
            half=config.USE_FP16,
            max_age=210, # MEMORY: Remember an object for 210 frames (approx 7 sec) even if hidden.
            conf_thres=config.CONF_THRESH,
            iou_thres=config.IOU_THRESH
        )

    def update(self, person_dets, chair_dets, frame):
        """
        Merges detections from both models and updates the tracker.
        
        Inputs:
          person_dets: Boxes of people [x,y,x,y,conf,0]
          chair_dets: Boxes of chairs [x,y,x,y,conf,56]
          frame: The video frame (needed for ReID to see colors/textures)
          
        Returns:
          tracks: Final list of tracked objects [x,y,x,y, ID, conf, class, ...]
        """
        # 1. Merge the lists (People + Chairs)
        dets_list = []
        
        if len(person_dets) > 0:
            dets_list.append(person_dets)
            
        if len(chair_dets) > 0:
            dets_list.append(chair_dets)
            
        # If the room is completely empty (no people, no chairs), stop here.
        if not dets_list:
            return np.empty((0, 8))
            
        # Combine into one big list for the tracker to process at once
        all_dets = np.vstack(dets_list)

        # 2. Update the Tracker
        # The tracker looks at the boxes AND the image pixels (ReID) to assign IDs.
        tracks = self.tracker.update(all_dets, frame)
        
        return tracks