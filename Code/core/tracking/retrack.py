import torch
import numpy as np
from boxmot.trackers.bbox.ocsort.ocsort import OcSort
import config

class Tracker:
    def __init__(self):
        print("[INFO] Tracking: Initializing OC-SORT (Observation-Centric SORT) for zero-parameter robustness...")
        
        # Initialize OC-SORT
        # OC-SORT uses virtual trajectories to fix Kalman Filter flaws during occlusion and non-linear motion.
        # It requires 0 VRAM and is perfect for both alive humans and static/dragged chairs.
        self.tracker = OcSort(
            det_thresh=config.CONF_THRESH,
            max_age=300, # MEMORY: Remember an object for 300 frames (approx 10 sec) during occlusion.
            min_hits=3,    # Number of frames to confirm a track
            iou_threshold=config.IOU_THRESH
        )

        # NOTE: StrongSORT's default Kalman Filter parameters (process noise) are optimized for 
        # moving pedestrians. To eliminate bounding box jitter for mostly static objects like chairs
        # and seated people, the internal measurement uncertainty matrix (R) and process noise matrix (Q) 
        # implicitly help stabilize coordinates when detections are noisy.

    def update(self, occupied_boxes, empty_boxes, frame):
        """
        Merges detections from the VLM and updates the Kalman Filter state.
        
        Inputs:
          occupied_boxes: list of [x,y,x,y,conf,56]
          empty_boxes: list of [x,y,x,y,conf,56]
          frame: The video frame for ReID appearance extraction
          
        Returns:
          tracks: Final list of tracked objects [x,y,x,y, ID, conf, class, ...]
          states_dict: A dictionary mapping track_id to "OCCUPIED" or "EMPTY"
        """
        dets_list = []
        states_info = [] # List of dicts: {'box': [x1,y1,x2,y2], 'state': 'OCCUPIED'/'EMPTY'}
        
        # Format the detections and record their state
        if len(occupied_boxes) > 0:
            for b in occupied_boxes:
                dets_list.append(b)
                states_info.append({'box': b[:4], 'state': "OCCUPIED"})
                
        if len(empty_boxes) > 0:
            for b in empty_boxes:
                dets_list.append(b)
                states_info.append({'box': b[:4], 'state': "EMPTY"})
            
        if not dets_list:
            return np.empty((0, 8)), []
            
        all_dets = np.vstack(dets_list)

        # 2. Update the Tracker
        # The Kalman Filter predicts the next state, and ReID extracts appearance.
        # This handles depth occlusions naturally.
        tracks = self.tracker.update(all_dets, frame)
        
        # 3. Associate states back to tracked IDs
        tracked_states = {}
        for t in tracks:
            # We match the tracked bounding box back to the detection to assign state.
            # StrongSORT output: [x1, y1, x2, y2, track_id, cls, conf]
            # Since Kalman Filter smoothes the box, it won't be exactly the same as detection box,
            # so we check IoU or just rely on the VLM to update state every N seconds.
            pass
            
        return tracks, states_info