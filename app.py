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
    
    kpi_html = f"""
    <div style="display: flex; gap: 1rem; margin-top: 1rem; font-family: sans-serif;">
        <div style="background: #1f2937; padding: 1.5rem; border-radius: 0.5rem; flex: 1; text-align: center; border: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Max Occupancy</div>
            <div style="font-size: 2rem; font-weight: bold; margin-top: 0.5rem; color: #ef4444; text-shadow: 0 0 15px rgba(239,68,68,0.8);">{max_occupancy}</div>
        </div>
        <div style="background: #1f2937; padding: 1.5rem; border-radius: 0.5rem; flex: 1; text-align: center; border: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Total Frames Processed</div>
            <div style="font-size: 2rem; font-weight: bold; margin-top: 0.5rem; color: #f8fafc;">{frame_count}</div>
        </div>
        <div style="background: #1f2937; padding: 1.5rem; border-radius: 0.5rem; flex: 1; text-align: center; border: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
            <div style="color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Engine Speed</div>
            <div style="font-size: 2rem; font-weight: bold; margin-top: 0.5rem; color: #3b82f6; text-shadow: 0 0 15px rgba(59,130,246,0.6);">{avg_fps:.1f} FPS</div>
        </div>
    </div>
    """
               
    return output_path, kpi_html


# --- SPECTRAVISION ENTERPRISE UI (GRADIO OVERHAUL) ---

custom_css = """
body, .gradio-container {
    background-color: #0B1120 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.panel, .block {
    background-color: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
}
button.primary {
    background-color: #0033CC !important;
    border-color: #0033CC !important;
    font-weight: bold !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    background-color: #002299 !important;
    box-shadow: 0 0 15px rgba(0, 51, 204, 0.6) !important;
}
h1, h2, h3, p, span {
    color: #e2e8f0 !important;
}
.jio-blue { color: #0033CC !important; }
.header-bar {
    background-color: #111827;
    padding: 20px;
    border-bottom: 1px solid #374151;
    margin-bottom: 20px;
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
}
"""

with gr.Blocks(title="Jio SpectraVision | Operations Overwatch", css=custom_css, theme=gr.themes.Base()) as interface:
    
    # Custom HTML Header matching the Flask dashboard
    gr.HTML("""
    <div class="header-bar">
        <div>
            <h1 style="margin: 0; font-size: 1.5rem; font-weight: bold; letter-spacing: 0.1em; color: white;">
                SPECTRA<span class="jio-blue">VISION</span> 
                <span style="font-size: 0.875rem; font-weight: normal; color: #9ca3af; margin-left: 8px;">| Operations Overwatch (VLM Edition)</span>
            </h1>
        </div>
        <div style="font-family: monospace; color: #9ca3af; font-size: 0.875rem;">
            BATCH ANALYTICS ENGINE
        </div>
    </div>
    """)
    
    gr.Markdown("Upload a CCTV clip (max 20 seconds) to process it using **LocateAnything-3B** and **OC-SORT tracking**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input CCTV Video", height=400)
            process_btn = gr.Button("🚀 Run Analytics Pipeline", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            video_output = gr.Video(label="Processed Output Stream", height=400, interactive=False)
    
    with gr.Row():
        kpi_output = gr.HTML("""
        <div style="text-align: center; padding: 2rem; border: 1px dashed #374151; border-radius: 0.5rem; color: #64748b; margin-top: 1rem;">
            📊 KPIs will appear here after the engine completes processing.
        </div>
        """)
            
    process_btn.click(
        fn=process_video_gpu, 
        inputs=video_input, 
        outputs=[video_output, kpi_output]
    )

if __name__ == "__main__":
    # share=True creates a free, temporary public URL (e.g., https://xxxx.gradio.live) 
    # that tunnels directly to your local GPU!
    interface.launch(share=True)
