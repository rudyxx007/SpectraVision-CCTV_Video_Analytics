import cv2
import sys
import time
import numpy as np
from pathlib import Path
import threading
import webbrowser
from queue import Queue, Empty
from scipy.optimize import linear_sum_assignment

# --- Setup Path ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config
from core.vision.preprocess import FramePreprocessor
from core.vision.vlm_detector import VLMDetector
from core.tracking.retrack import Tracker
from core.utils.logger import AnalyticsLogger
import dashboard.app as dashboard_app

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0.0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

class AsyncVideoCapture:
    """Asynchronous VideoCapture to simulate RTSP buffer lag and dropped frames."""
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.q = Queue(maxsize=3) # Small buffer to drop old frames
        self.stopped = False
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        
    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)
            success, frame = self.cap.read()
            if not success:
                # Loop the video if it reaches the end (simulating endless CCTV)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Simulate real-time 30 FPS (since local files read instantly)
            time.sleep(1/30.0)
            
            # Clear queue if full to always keep the latest RTSP frame
            if self.q.full():
                try:
                    self.q.get_nowait()
                except Empty:
                    pass
            self.q.put(frame)
            
    def read(self):
        try:
            return True, self.q.get(timeout=1.0)
        except Empty:
            return False, None
            
    def release(self):
        self.stopped = True
        self.cap.release()

def run_analytics():
    print("[INFO] Starting Operations Overwatch (VLM + Kalman Filter Edition)...")
    
    # 1. Init Modules
    try:
        preprocessor = FramePreprocessor()
        tracker = Tracker()
        logger = AnalyticsLogger()
        vlm_detector = VLMDetector()
    except Exception as e:
        print(f"[ERROR] Init Failed: {e}")
        return

    # 2. Async RTSP Video Setup
    print(f"[INFO] Connecting to RTSP Stream: {config.VIDEO_SOURCE}")
    async_cap = AsyncVideoCapture(config.VIDEO_SOURCE)
    
    W, H = 1280, 736
    heatmap_accum = np.zeros((H, W), dtype=np.float32)

    # --- SERVER BOOT ---
    print("[INFO] Starting UI Server in background...")
    def run_flask():
        dashboard_app.app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    print("[INFO] Opening Enterprise Dashboard...")
    # webbrowser.open("http://127.0.0.1:5000") # Optional
    # ---------------------------------
    
    window_name = "CCTV Video Analytics (VLM + KF)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_vlm_time = time.time()
    current_chair_states = {}
    
    # Shared variables for VLM Thread
    vlm_lock = threading.Lock()
    vlm_results = {"occ": [], "emp": []}
    vlm_is_running = False

    def vlm_worker(frame):
        nonlocal vlm_results, vlm_is_running
        try:
            occ, emp = vlm_detector.detect_chairs(frame)
            with vlm_lock:
                vlm_results = {"occ": occ, "emp": emp}
        except Exception as e:
            print(f"[ERROR] VLM Inference failed: {e}")
        finally:
            vlm_is_running = False

    while True:
        loop_start = time.time()

        success, raw_frame = async_cap.read()
        if not success or raw_frame is None:
            continue
            
        frame_count += 1
        frame_resized = cv2.resize(raw_frame, (W, H))
        clean_frame = preprocessor.process(frame_resized)
        
        # --- PHASE 1: Asynchronous VLM Inference ---
        # Run LocateAnything-3B every N seconds in background
        if (time.time() - last_vlm_time) > config.VLM_INTERVAL and not vlm_is_running:
            vlm_is_running = True
            last_vlm_time = time.time()
            threading.Thread(target=vlm_worker, args=(clean_frame.copy(),), daemon=True).start()

        # Grab latest VLM boxes safely
        occ_boxes, emp_boxes = [], []
        with vlm_lock:
            occ_boxes = vlm_results["occ"]
            emp_boxes = vlm_results["emp"]
            # Consume boxes so StrongSORT doesn't keep updating on stale detections
            # The Kalman filter will simply *predict* the next state if no new detections arrive!
            vlm_results = {"occ": [], "emp": []}

        # --- PHASE 2: Kalman Filter Tracking (StrongSORT) ---
        # When occ_boxes/emp_boxes are empty, StrongSORT enters the "Predict" phase exclusively.
        # This resolves depth occlusions and eliminates bounding box jitter completely.
        tracks, state_hints = tracker.update(occ_boxes, emp_boxes, clean_frame)
        
        # --- PHASE 3: State Assignment ---
        if len(tracks) > 0 and len(state_hints) > 0:
            # Compute Cost Matrix (1 - IoU)
            cost_matrix = np.ones((len(tracks), len(state_hints)))
            for t_idx, track in enumerate(tracks):
                track_box = track[:4]
                for s_idx, hint in enumerate(state_hints):
                    hint_box = hint['box']
                    iou = compute_iou(track_box, hint_box)
                    cost_matrix[t_idx, s_idx] = 1.0 - iou
            
            # Hungarian Algorithm Matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for t_idx, s_idx in zip(row_ind, col_ind):
                # Only accept matches with reasonable overlap (e.g. IoU > 0.3 -> Cost < 0.7)
                if cost_matrix[t_idx, s_idx] < 0.7:
                    track_id = int(tracks[t_idx][4])
                    state = state_hints[s_idx]['state']
                    current_chair_states[track_id] = state
            
        # FPS Calculation
        fps = 1.0 / (time.time() - loop_start + 1e-6)

        # Analytics
        logger.log_frame(frame_count, current_chair_states, current_fps=fps)

        # Visualization
        vis_frame = clean_frame.copy()

        # Heatmap
        if dashboard_app.HEATMAP_ACTIVE:
            smoothed = cv2.GaussianBlur(heatmap_accum, (51, 51), 0)
            max_val = np.max(smoothed)
            if max_val > 0:
                normed = (smoothed / max_val * 255).astype(np.uint8)
                color_map = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
                mask = normed > 15
                blended = cv2.addWeighted(vis_frame, 0.5, color_map, 0.5, 0)
                vis_frame[mask] = blended[mask]

        # Draw Tracks smoothed by Kalman Filter
        for t in tracks:
            if len(t) < 5: continue
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"ID-{tid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send to Dashboard
        dashboard_app.update_video_frame(vis_frame)
        
        cv2.imshow(window_name, vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    async_cap.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] Processing Finished.")

if __name__ == "__main__":
    run_analytics()