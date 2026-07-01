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
    
    # Cap at 60 seconds to allow longer video processing
    max_frames = int(60 * fps)
    
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
    <div style="display: flex; gap: 1.5rem; margin-top: 1rem; font-family: 'Rajdhani', sans-serif;">
        <div class="kpi-card" style="flex: 1; padding: 1.5rem; border-radius: 12px; background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; w-full; height: 3px; background: linear-gradient(90deg, transparent, #ef4444, transparent); width: 100%;"></div>
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600;">Max Occupancy</div>
            <div style="font-size: 3rem; font-weight: 700; margin-top: 0.5rem; color: #ef4444; text-shadow: 0 0 20px rgba(239,68,68,0.4); line-height: 1;">{max_occupancy}</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">Detected Objects</div>
        </div>
        
        <div class="kpi-card" style="flex: 1; padding: 1.5rem; border-radius: 12px; background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; w-full; height: 3px; background: linear-gradient(90deg, transparent, #3b82f6, transparent); width: 100%;"></div>
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600;">Frames Processed</div>
            <div style="font-size: 3rem; font-weight: 700; margin-top: 0.5rem; color: #f8fafc; text-shadow: 0 0 20px rgba(255,255,255,0.2); line-height: 1;">{frame_count}</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">Total Batches</div>
        </div>
        
        <div class="kpi-card" style="flex: 1; padding: 1.5rem; border-radius: 12px; background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; w-full; height: 3px; background: linear-gradient(90deg, transparent, #10b981, transparent); width: 100%;"></div>
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600;">Inference Speed</div>
            <div style="font-size: 3rem; font-weight: 700; margin-top: 0.5rem; color: #10b981; text-shadow: 0 0 20px rgba(16,185,129,0.4); line-height: 1;">{avg_fps:.1f}</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">Frames per second</div>
        </div>
    </div>
    """
               
    return output_path, kpi_html


# --- ULTRA-PREMIUM SAAS UI OVERHAUL ---

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Rajdhani:wght@500;600;700&display=swap');

/* Base Theme & Hide Gradio Chrome */
body, .gradio-container {
    background: #020617 !important; /* Tailwind Slate-950 */
    background-image: 
        radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(225,39%,30%,0.1) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(339,49%,30%,0.05) 0, transparent 50%) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
footer { display: none !important; } /* Hide Built with Gradio */

/* Glassmorphism Header */
.glass-header {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding: 1.5rem 2rem;
    margin: -20px -20px 2rem -20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
}

.title-glow {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    margin: 0;
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-accent { color: #3b82f6; -webkit-text-fill-color: #3b82f6; text-shadow: 0 0 20px rgba(59,130,246,0.5); }

/* Component Styling */
.panel, .block {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.2) !important;
}

/* Run Button */
button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 1rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 0 15px rgba(37, 99, 235, 0.3) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.5) !important;
}

/* Video Player Tweaks */
video { border-radius: 8px !important; }
"""

# Use a purely dark base theme to override gradio's defaults
theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="Jio SpectraVision", css=custom_css, theme=theme) as interface:
    
    # Ultra-Premium Header
    gr.HTML("""
    <div class="glass-header">
        <h1 class="title-glow">SPECTRA<span class="brand-accent">VISION</span></h1>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #10b981; box-shadow: 0 0 10px #10b981; animation: pulse 2s infinite;"></div>
            <span style="font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">A100 Cloud Node Active</span>
        </div>
    </div>
    <style>@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }</style>
    """)
    
    # Dashboard Layout (Sidebar + Main Content)
    with gr.Row(style={"gap": "2rem"}):
        
        # LEFT COLUMN (Controls)
        with gr.Column(scale=3):
            gr.HTML("""
            <div style="margin-bottom: 1.5rem;">
                <h2 style="font-family: 'Rajdhani', sans-serif; color: #f8fafc; font-size: 1.25rem; font-weight: 600; letter-spacing: 0.05em; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem;">COMMAND CENTER</h2>
                <p style="color: #64748b; font-size: 0.85rem; line-height: 1.5; margin-top: 0.5rem;">Upload CCTV footage to the cloud ingest pipeline. The VLM engine will automatically detect and track entities using Zero-Shot spatial reasoning.</p>
            </div>
            """)
            
            # format="mp4" forces Gradio to transcode webcam webm/av1 recordings into standard MP4 for OpenCV
            video_input = gr.Video(label="Source Footage", height=320, format="mp4", elem_classes=["panel"])
            
            gr.HTML("<div style='height: 1rem;'></div>") # Spacer
            
            process_btn = gr.Button("INITIALIZE ANALYTICS", variant="primary")
        
        # RIGHT COLUMN (Telemetry & Output)
        with gr.Column(scale=7):
            gr.HTML("""
            <h2 style="font-family: 'Rajdhani', sans-serif; color: #f8fafc; font-size: 1.25rem; font-weight: 600; letter-spacing: 0.05em; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem; margin-bottom: 1.5rem;">LIVE TELEMETRY</h2>
            """)
            
            video_output = gr.Video(label="Engine Output Stream", height=450, interactive=False, elem_classes=["panel"])
            
            kpi_output = gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; height: 120px; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; background: rgba(15, 23, 42, 0.3); margin-top: 1.5rem;">
                <span style="font-family: 'Rajdhani', sans-serif; color: #475569; font-size: 1.1rem; letter-spacing: 0.1em; text-transform: uppercase;">Awaiting Stream Data...</span>
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
