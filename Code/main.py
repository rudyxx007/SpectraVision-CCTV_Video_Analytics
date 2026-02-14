import cv2
import sys
import time
import numpy as np
from pathlib import Path
import threading
import webbrowser

# --- Setup Path ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config
from core.vision.preprocess import FramePreprocessor
from core.vision.detector import DualDetector
from core.vision.segmentor import SegmentationEngine 
from core.tracking.retrack import Tracker
from core.logic.pose import PoseEngine
from core.inference.fusion import FusionEngine
from core.utils.logger import AnalyticsLogger
from core.sgg.scene_graph import SceneGraphEngine 
import dashboard.app as dashboard_app

def run_analytics():
    print("[INFO] Starting Operations Overwatch...")
    
    # 1. Init Modules
    try:
        preprocessor = FramePreprocessor()
        detector = DualDetector()
        tracker = Tracker()
        pose_engine = PoseEngine()
        fusion_engine = FusionEngine()
        logger = AnalyticsLogger()
        
        segmentor = SegmentationEngine() 
        sgg_engine = SceneGraphEngine()
    except Exception as e:
        print(f"[ERROR] Init Failed: {e}")
        return

    # 2. Video Setup
    video_path = str(config.VIDEO_SOURCE)
    print(f"[INFO] Attempting to load video at: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] CRITICAL: OpenCV could not open the video file!")
        print(f"[ERROR] Please check if 'office_cctv.mp4' actually exists in: {config.DATA_DIR}\\input_video\\")
        return # Stop the script here
        
    W, H = 1280, 736
    
    save_path = config.LOGS_DIR.parent / "final_output.mp4"
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
    
    # Heatmap Accumulator Array (Starts completely blank)
    heatmap_accum = np.zeros((H, W), dtype=np.float32)

    # --- SERVER BOOT ---
    print("[INFO] Starting UI Server in background...")
    def run_flask():
        # use_reloader=False is mandatory when running Flask in a thread
        dashboard_app.app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    print("[INFO] Opening Enterprise Dashboard...")
    webbrowser.open("http://127.0.0.1:5000")
    # ---------------------------------
    
    # Optional debugging window
    window_name = "CCTV Video Analytics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_sgg_time = time.time()
    current_chair_states = {}

    while True:
        loop_start = time.time() # Start FPS clock

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
        interactions = pose_engine.check_interaction(people, chairs, p_boxes, kpts)
        
        for inter in interactions:
            pid = inter['person_id']
            cid = inter['chair_id']
            pose_state = inter['state'] 
            
            if cid != -1:
                score_pose = 1.0 if pose_state == "SITTING" else 0.0
                score_iou = inter['iou'] 
                score_sgg = None
                
                # Fetch tracks to pass bounding boxes to SAM/Qwen
                p_track = next((p for p in people if int(p[4]) == pid), None)
                c_track = next((c for c in chairs if int(c[4]) == cid), None)

                if p_track is not None and c_track is not None:
                    # Run SAM 3
                    if inter['iou'] > config.SAM_TRIGGER_IOU:
                         score_iou = segmentor.get_mask_iou(clean_frame, p_track[:4], c_track[:4])

                    # Run Qwen3-VL (Every 5 seconds)
                    if (time.time() - last_sgg_time) > config.SGG_INTERVAL and pose_state == "SITTING":
                        score_sgg = sgg_engine.verify_interaction(clean_frame, p_track[:4], c_track[:4])
                
                # Fusion
                final_state, conf = fusion_engine.update_state(cid, score_pose, score_iou, score_sgg)
                current_chair_states[cid] = final_state

                # HEATMAP BUILDER: Add heat for occupied chairs
                if final_state == "OCCUPIED" and c_track is not None:
                    cx1, cy1, cx2, cy2 = map(int, c_track[:4])
                    center_x, center_y = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                    cv2.circle(heatmap_accum, (center_x, center_y), 40, 1.0, -1)

        if (time.time() - last_sgg_time) > config.SGG_INTERVAL:
            last_sgg_time = time.time()

        # FPS Calculation
        fps = 1.0 / (time.time() - loop_start)

        # --- PHASE 3: Analytics ---
        logger.log_frame(frame_count, current_chair_states, current_fps=fps)

        # --- Visualization & Heatmap Render ---
        vis_frame = clean_frame.copy()

        # HEATMAP RENDERER (Triggered from UI)
        if dashboard_app.HEATMAP_ACTIVE:
            smoothed = cv2.GaussianBlur(heatmap_accum, (51, 51), 0)
            max_val = np.max(smoothed)
            if max_val > 0:
                normed = (smoothed / max_val * 255).astype(np.uint8)
                color_map = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
                mask = normed > 15
                vis_frame[mask] = cv2.addWeighted(vis_frame[mask], 0.5, color_map[mask], 0.5, 0)

        # Draw Base Bounding Boxes
        for c in chairs:
            x1, y1, x2, y2 = map(int, c[:4])
            cid = int(c[4])
            state = current_chair_states.get(cid, "EMPTY")
            color = (0, 0, 255) if state == "OCCUPIED" else (0, 255, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"C-{cid} [{state}]", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for p in people:
            x1, y1, x2, y2 = map(int, p[:4])
            pid = int(p[4])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(vis_frame, f"P-{pid}", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        out.write(vis_frame)
        cv2.imshow(window_name, vis_frame)
        
        # Send frame to Flask
        dashboard_app.update_video_frame(vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] Processing Finished.")

if __name__ == "__main__":
    run_analytics()