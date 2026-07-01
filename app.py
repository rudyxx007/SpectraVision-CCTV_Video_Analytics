import gradio as gr
import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import spaces

# Ensure Code directory is in path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT / "Code") not in sys.path:
    sys.path.append(str(ROOT / "Code"))

import config
from core.vision.preprocess import FramePreprocessor
from core.vision.vlm_detector import VLMDetector
from core.tracking.retrack import Tracker

# Global singletons
preprocessor = FramePreprocessor()
vlm_detector = VLMDetector()

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

@spaces.GPU
def process_video_gpu(input_video_path):
    if not input_video_path:
        return None, "Please upload a video."
        
    tracker = Tracker()
    cap = cv2.VideoCapture(input_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Cap at 20 seconds to prevent ZeroGPU timeouts
    max_frames = int(20 * fps)
    
    W, H = 1280, 736
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    frame_count = 0
    last_vlm_frame = -9999
    
    # Run VLM every config.VLM_INTERVAL seconds
    vlm_frame_interval = int(config.VLM_INTERVAL * fps)
    if vlm_frame_interval < 1: vlm_frame_interval = 1
    
    occ_boxes, emp_boxes = [], []
    current_states = {}
    
    max_occupancy = 0
    start_time = time.time()
    
    print("[INFO] Beginning Video Analytics on GPU...")
    
    while True:
        success, raw_frame = cap.read()
        if not success or frame_count >= max_frames:
            break
            
        frame_count += 1
        frame_resized = cv2.resize(raw_frame, (W, H))
        clean_frame = preprocessor.process(frame_resized)
        
        # Run VLM Inference
        if (frame_count - last_vlm_frame) >= vlm_frame_interval:
            occ_boxes, emp_boxes = vlm_detector.detect_chairs(clean_frame)
            last_vlm_frame = frame_count
            
        # Kalman Tracking
        tracks, state_hints = tracker.update(occ_boxes, emp_boxes, clean_frame)
        
        # State Assignment (Hungarian Algorithm)
        if len(tracks) > 0 and len(state_hints) > 0:
            cost_matrix = np.ones((len(tracks), len(state_hints)))
            for t_idx, track in enumerate(tracks):
                track_box = track[:4]
                for s_idx, hint in enumerate(state_hints):
                    hint_box = hint['box']
                    iou = compute_iou(track_box, hint_box)
                    cost_matrix[t_idx, s_idx] = 1.0 - iou
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for t_idx, s_idx in zip(row_ind, col_ind):
                if cost_matrix[t_idx, s_idx] < 0.7:
                    track_id = int(tracks[t_idx][4])
                    state = state_hints[s_idx]['state']
                    current_states[track_id] = state
                    
        # Calculate Occupancy KPI
        occupancy_count = sum(1 for state in current_states.values() if state == "OCCUPIED")
        if occupancy_count > max_occupancy:
            max_occupancy = occupancy_count
            
        vis_frame = clean_frame.copy()
        
        # Draw Tracks
        for t in tracks:
            if len(t) < 5: continue
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            
            state = current_states.get(tid, "EMPTY")
            color = (0, 0, 255) if state == "OCCUPIED" else (0, 255, 0) # Red if occupied, Green if available
            label = f"ID-{tid} [{state}]"
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        out.write(vis_frame)
        
        # Clear VLM boxes so Kalman predicts smoothly
        occ_boxes, emp_boxes = [], []
        
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0
    
    print(f"[SUCCESS] Processed {frame_count} frames at {avg_fps:.1f} FPS.")
    
    kpi_text = (
        f"### Analytics Summary\n"
        f"- **Max Detected Occupancy:** {max_occupancy} objects\n"
        f"- **Frames Processed:** {frame_count}\n"
        f"- **Engine Processing Speed:** {avg_fps:.1f} FPS"
    )
               
    return output_path, kpi_text

# --- Gradio UI ---
with gr.Blocks(title="Operations Overwatch (CCTV Video Analytics)", theme=gr.themes.Base()) as interface:
    gr.Markdown("<div align='center'><h1>👁️ Operations Overwatch</h1><p>Autonomous VLM-Powered Surveillance Engine</p></div>")
    gr.Markdown("Upload a CCTV clip (max 20 seconds) to process it using **LocateAnything-3B** and **OC-SORT tracking**.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input CCTV Video")
            process_btn = gr.Button("Run Analytics Pipeline", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Processed Output Stream")
            kpi_output = gr.Markdown("📊 **KPIs will appear here after processing.**")
            
    process_btn.click(
        fn=process_video_gpu, 
        inputs=video_input, 
        outputs=[video_output, kpi_output]
    )

if __name__ == "__main__":
    # share=True creates a free, temporary public URL (e.g., https://xxxx.gradio.live) 
    # that tunnels directly to your local GPU!
    interface.launch(share=True)
