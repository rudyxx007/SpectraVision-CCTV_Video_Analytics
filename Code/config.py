import torch
from pathlib import Path

# ==================================================
# 📂 PATHS & DIRECTORIES
# ==================================================
# Get the root directory of the project (D:/Jio/CCTV-Video-Analytics/)
BASE_DIR = Path(__file__).resolve().parent

# Weights Folder
WEIGHTS_DIR = BASE_DIR / "weights"

# Data Folders
INPUT_VIDEO_DIR = BASE_DIR / "data/input_video"
OUTPUT_LOGS_DIR = BASE_DIR / "data/logs"
MASK_DIR = BASE_DIR / "data/masks"

# Specific File Paths
# 1. Video Source (Replace with 0 for Webcam or RTSP link for IP Camera)
VIDEO_SOURCE = INPUT_VIDEO_DIR / "office_cctv.mp4"

# 2. ROI Mask (Optional - Black and white image defining the active area)
# If None, no masking is applied.
ROI_MASK_PATH = MASK_DIR / "room_mask.png" 

# ==================================================
# 🧠 MODEL WEIGHTS
# ==================================================
# Model 1: Pose Estimation (Detects People + Skeletons)
POSE_MODEL_WEIGHTS = WEIGHTS_DIR / "yolo26s-pose.pt"

# Model 2: Object Detection (Detects Chairs/Furniture)
# We use 'n' (nano) for speed. It auto-downloads if not found.
CHAIR_MODEL_WEIGHTS = "yolo26n.pt" 

# Model 3: Re-Identification (Keeps IDs stable)
REID_WEIGHTS = WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"

# ==================================================
# ⚙️ HARDWARE SETTINGS
# ==================================================
# "0" for the first GPU, "cpu" if no GPU found
DEVICE = "0" if torch.cuda.is_available() else "cpu"

# Use FP16 (Half Precision) to speed up inference on RTX 5050
USE_FP16 = True

# ==================================================
# 🎛️ AI TUNING PARAMETERS
# ==================================================
# Confidence: Ignore detections below 40% certainty
CONF_THRESH = 0.4 

# NMS IoU: Merge boxes that overlap by more than 50%
IOU_THRESH = 0.5

# ==================================================
# 🖼️ PREPROCESSING SETTINGS
# ==================================================
# Enable Contrast Enhancement (Good for dark CCTV)
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0  # Strength of contrast (1.0 = low, 4.0 = high)
CLAHE_GRID_SIZE = (8, 8) # Size of the local grid for equalization

# Enable Noise Reduction
ENABLE_MEDIAN_BLUR = True
BLUR_KERNEL_SIZE = 3 # Must be an odd number (3, 5, 7). Higher = More blur.