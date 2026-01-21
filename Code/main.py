import cv2
import sys
import numpy as np
from pathlib import Path
import time

# --- Setup Python Path ---
# This ensures Python can find the 'core' folder
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config
from core.vision.preprocess import FramePreprocessor
from core.vision.detector import DualDetector
from core.tracking.retrack import Tracker

def run_analytics():
    # ==================================================
    # 1. INITIALIZATION
    # ==================================================
    print("[INFO] Starting Jio Analytics System...")
    
    # Initialize Core Modules
    try:
        preprocessor = FramePreprocessor()
        detector = DualDetector()
        tracker = Tracker()
    except Exception as e:
        print(f"[ERROR] Module Initialization Failed: {e}")
        return

    # Open Video Source
    cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {config.VIDEO_SOURCE}")
        return

    # Get Video Properties (for optional saving later)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video Loaded: {width}x{height} @ {fps} FPS")

    # ==================================================
    # 2. MAIN LOOP
    # ==================================================
    frame_count = 0
    
    while True:
        start_time = time.time()
        
        # Read Frame
        success, frame = cap.read()
        if not success:
            print("[INFO] End of video stream.")
            break
        
        frame_count += 1

        # --- STEP A: PREPROCESSING ---
        # Clean the image (Blur, CLAHE, Mask)
        clean_frame = preprocessor.process(frame)

        # --- STEP B: PERCEPTION (VISION) ---
        # Get raw detections from the AI models
        person_boxes, chair_boxes, keypoints = detector.detect(clean_frame)

        # --- STEP C: TRACKING (MEMORY) ---
        # Assign IDs to the boxes
        # tracks format: [x1, y1, x2, y2, id, conf, class_id, ...]
        tracks = tracker.update(person_boxes, chair_boxes, clean_frame)

        # --- STEP D: VISUALIZATION (TASK 2 OUTPUT) ---
        # Draw the results on the ORIGINAL frame (not the preprocessed one)
        for track in tracks:
            # Extract coordinates and IDs
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            class_id = int(track[6])

            # Define Colors & Labels
            if class_id == 0: 
                # Person = Green
                color = (0, 255, 0) 
                label = f"P-{track_id}"
            elif class_id == 56: 
                # Chair = Red
                color = (0, 0, 255) 
                label = f"C-{track_id}"
            else:
                continue # Skip other objects

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background (for readability)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw Text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Calculate FPS for performance monitoring
        process_time = time.time() - start_time
        current_fps = 1 / process_time if process_time > 0 else 0
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show Output
        cv2.imshow("Jio Analytics - Task 2 Baseline", frame)

        # Exit on 'Q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    # ==================================================
    # 3. CLEANUP
    # ==================================================
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System Stopped.")

if __name__ == "__main__":
    run_analytics()