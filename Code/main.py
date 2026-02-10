import cv2
import sys
import time
import numpy as np
from pathlib import Path

# --- Setup Path ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config
from core.vision.preprocess import FramePreprocessor
from core.vision.detector import DualDetector
# from core.vision.segmentor import SegmentationEngine # Uncomment if SAM weights exist
from core.tracking.retrack import Tracker
from core.logic.pose import PoseEngine
from core.logic.fusion import FusionEngine
from core.utils.logger import AnalyticsLogger
# from core.sgg.scene_graph import SceneGraphEngine # Uncomment if VLM weights exist

def run_analytics():
    print("[INFO] Starting CCTV Analytics (Phase 1-3 Complete)...")
    
    # 1. Init Modules
    try:
        preprocessor = FramePreprocessor()
        detector = DualDetector()
        tracker = Tracker()
        pose_engine = PoseEngine()
        fusion_engine = FusionEngine()
        logger = AnalyticsLogger()
        
        # Optional Heavy Modules (Task 4 & 5)
        # segmentor = SegmentationEngine() 
        # sgg_engine = SceneGraphEngine()
    except Exception as e:
        print(f"[ERROR] Init Failed: {e}")
        return

    # 2. Video Setup
    cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
    W, H = 1280, 736
    
    # Save Output
    save_path = config.LOGS_DIR.parent / "final_output.mp4"
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
    
    window_name = "CCTV Video Analytics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    last_sgg_time = time.time()

    while True:
        start_time = time.time()
        success, raw_frame = cap.read()
        if not success: break
        frame_count += 1
        
        # --- PHASE 1: Perception ---
        frame_resized = cv2.resize(raw_frame, (W, H))
        clean_frame = preprocessor.process(frame_resized)
        
        # Detect & Track
        p_boxes, c_boxes, kpts = detector.detect(clean_frame)
        tracks = tracker.update(p_boxes, c_boxes, clean_frame)
        
        # Split Tracks
        people = [t for t in tracks if int(t[6]) == 0]
        chairs = [t for t in tracks if int(t[6]) == 56]
        
        # --- PHASE 2: Verification ---
        current_chair_states = {}
        
        # Get Pose Interactions
        # We use the PoseEngine to get a base "Sitting" status based on geometry
        interactions = pose_engine.check_interaction(people, chairs, kpts)
        
        for inter in interactions:
            pid = inter['person_id']
            cid = inter['chair_id']
            pose_state = inter['state'] # SITTING / STANDING / TOUCHING
            
            if cid != -1:
                # 1. Pose Score
                score_pose = 1.0 if pose_state == "SITTING" else 0.0
                
                # 2. IoU Score (We already calculated this in pose check, reusing)
                # For simplicity here, we assume if interacting, high IoU.
                score_iou = 0.8 
                
                # 3. SGG Score (Task 5) - Run every 5 seconds
                score_sgg = None
                if (time.time() - last_sgg_time) > config.SGG_INTERVAL:
                    # Trigger VLM logic here if enabled
                    # score_sgg = sgg_engine.verify_interaction(...)
                    pass
                
                # 4. Fusion (Task 6)
                final_state, conf = fusion_engine.update_state(cid, score_pose, score_iou, score_sgg)
                current_chair_states[cid] = final_state

        # Update global SGG timer
        if (time.time() - last_sgg_time) > config.SGG_INTERVAL:
            last_sgg_time = time.time()

        # --- PHASE 3: Analytics ---
        logger.log_frame(frame_count, current_chair_states)

        # --- Visualization ---
        vis_frame = clean_frame.copy()
        
        # Draw Chairs
        for c in chairs:
            x1, y1, x2, y2 = map(int, c[:4])
            cid = int(c[4])
            state = current_chair_states.get(cid, "EMPTY")
            color = (0, 0, 255) if state == "OCCUPIED" else (0, 255, 0) # Red if taken, Green if free
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"C-{cid} [{state}]", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw People
        for p in people:
            x1, y1, x2, y2 = map(int, p[:4])
            pid = int(p[4])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(vis_frame, f"P-{pid}", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display
        out.write(vis_frame)
        cv2.imshow(window_name, vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] Processing Finished.")

if __name__ == "__main__":
    run_analytics()