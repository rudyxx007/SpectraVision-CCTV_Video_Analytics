import numpy as np

class PoseEngine:
    def __init__(self):
        print("[INFO] Pose Engine: Initialized (Smart Occlusion Logic).")
        self.IOU_THRESH = 0.15      
        self.KNEE_ANGLE_THRESH = 110 # Degrees. < 110 typically indicates sitting.

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def calculate_angle(self, a, b, c):
        """Calculates the angle at the knee (b) given hip (a) and ankle (c)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def check_interaction(self, person_tracks, chair_tracks, keypoints):
        """
        Inputs:
            person_tracks: List of [x1, y1, x2, y2, ID, conf, class]
            chair_tracks:  List of [x1, y1, x2, y2, ID, conf, class]
            keypoints:     Raw keypoints from detector
        """
        interactions = []
        for i, p_track in enumerate(person_tracks):
            p_box = p_track[:4]
            p_id = int(p_track[4])
            
            best_iou = 0
            best_chair_id = -1
            
            # Check overlap with every chair track
            for c_track in chair_tracks:
                iou = self.calculate_iou(p_box, c_track[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_chair_id = int(c_track[4]) # Capture the Unique Chair ID
            
            state = "STANDING"
            
            # Logic: If Touching -> Check Legs or Occlusion
            if best_iou > self.IOU_THRESH:
                if i < len(keypoints): 
                    kpts = keypoints[i] 
                    # Right leg indices: Hip=12, Knee=14, Ankle=16
                    # Left leg indices: Hip=11, Knee=13, Ankle=15
                    
                    # Check visibility (conf > 0.5)
                    r_vis = kpts[12][2] > 0.5 and kpts[14][2] > 0.5 and kpts[16][2] > 0.5
                    l_vis = kpts[11][2] > 0.5 and kpts[13][2] > 0.5 and kpts[15][2] > 0.5

                    # CASE A: Legs are Visible -> Use Geometric Math
                    if r_vis or l_vis:
                        angle_r = self.calculate_angle(kpts[12][:2], kpts[14][:2], kpts[16][:2]) if r_vis else 180
                        angle_l = self.calculate_angle(kpts[11][:2], kpts[13][:2], kpts[15][:2]) if l_vis else 180
                        
                        if angle_r < self.KNEE_ANGLE_THRESH or angle_l < self.KNEE_ANGLE_THRESH:
                            state = "SITTING"
                        else:
                            state = "TOUCHING" # Legs straight but touching chair
                    
                    # CASE B: Legs are Occluded (Desk) -> Assume Sitting
                    else:
                        # High overlap + hidden legs = Sitting at desk
                        state = "SITTING"
                else:
                    # CASE C: No skeleton data at all -> Fallback to overlap
                    state = "SITTING"

            interactions.append({
                "person_id": p_id, 
                "chair_id": best_chair_id, 
                "state": state
            })
            
        return interactions