import numpy as np

class PoseEngine:
    def __init__(self):
        print("[INFO] Pose Engine: Initialized (IoU Tracking + Skeleton Logic).")
        self.IOU_THRESH = 0.15      
        self.KNEE_ANGLE_THRESH = 110 

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def check_interaction(self, person_tracks, chair_tracks, raw_person_boxes, keypoints):
        interactions = []
        for p_track in person_tracks:
            p_box = p_track[:4]
            p_id = int(p_track[4])
            
            # --- FIX: Match tracked box to raw detection keypoints via IoU ---
            kpts = None
            best_kpt_iou = 0
            for j, raw_box in enumerate(raw_person_boxes):
                iou = self.calculate_iou(p_box, raw_box[:4])
                if iou > best_kpt_iou and iou > 0.5: # Must overlap significantly
                    best_kpt_iou = iou
                    if j < len(keypoints):
                        kpts = keypoints[j]
            
            best_chair_iou = 0
            best_chair_id = -1
            
            for c_track in chair_tracks:
                iou = self.calculate_iou(p_box, c_track[:4])
                if iou > best_chair_iou:
                    best_chair_iou = iou
                    best_chair_id = int(c_track[4])
            
            state = "STANDING"
            
            if best_chair_iou > self.IOU_THRESH:
                if kpts is not None: 
                    # Right: 12, 14, 16 | Left: 11, 13, 15
                    r_vis = kpts[12][2] > 0.5 and kpts[14][2] > 0.5 and kpts[16][2] > 0.5
                    l_vis = kpts[11][2] > 0.5 and kpts[13][2] > 0.5 and kpts[15][2] > 0.5

                    if r_vis or l_vis:
                        angle_r = self.calculate_angle(kpts[12][:2], kpts[14][:2], kpts[16][:2]) if r_vis else 180
                        angle_l = self.calculate_angle(kpts[11][:2], kpts[13][:2], kpts[15][:2]) if l_vis else 180
                        
                        if angle_r < self.KNEE_ANGLE_THRESH or angle_l < self.KNEE_ANGLE_THRESH:
                            state = "SITTING"
                        else:
                            state = "TOUCHING" 
                    else:
                        state = "SITTING" # Legs Occluded (Desk)
                else:
                    state = "SITTING" # No skeleton, default to sitting on overlap

            interactions.append({"person_id": p_id, "chair_id": best_chair_id, "state": state, "iou": best_chair_iou})
            
        return interactions