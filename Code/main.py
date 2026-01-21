import cv2
import sys
from pathlib import Path

# Add project root to path so we can import 'core'
sys.path.append(str(Path(__file__).parent))

import config
from core.vision.detector import DualDetector
from core.tracking.retrack import Tracker

def run_pipeline():
    # 1. Init Modules
    detector = DualDetector()
    tracker = Tracker()

    # 2. Open Video
    cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    print("🚀 Pipeline Started. Press 'Q' to exit.")

    while True:
        success, frame = cap.read()
        if not success: break

        # --- PHASE 1: VISION ---
        p_boxes, c_boxes, kpts = detector.detect(frame)

        # --- PHASE 2: TRACKING ---
        tracks = tracker.update(p_boxes, c_boxes, frame)

        # --- PHASE 3: VISUALIZATION (Temp) ---
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            cls = int(t[6])

            if cls == 0: # Person
                color = (0, 255, 0) # Green
                label = f"P-{tid}"
            else: # Chair
                color = (0, 0, 255) # Red
                label = f"C-{tid}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 0, 0.6, color, 2)

        cv2.imshow("Jio Analytics 2026", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()