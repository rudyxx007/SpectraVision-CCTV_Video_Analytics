import torch
from pathlib import Path

# ==================================================
# PATHS & DIRECTORIES
# ==================================================
# This file is in Code/config.py
CODE_DIR = Path(__file__).resolve().parent

# The data folder is inside the Code folder
DATA_DIR = CODE_DIR / "data"

# The weights folder is one level up in the project root
PROJECT_ROOT = CODE_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Subfolders for Data
INPUT_VIDEO_DIR = DATA_DIR / "input_video"
OUTPUT_LOGS_DIR = DATA_DIR / "logs"
MASK_DIR = DATA_DIR / "masks"

# Specific File Paths
VIDEO_SOURCE = INPUT_VIDEO_DIR / "office_cctv.mp4"
ROI_MASK_PATH = MASK_DIR / "room_mask.png" 

# ==================================================
# MODEL WEIGHTS
# ==================================================
POSE_MODEL_WEIGHTS = WEIGHTS_DIR / "yolo26s-pose.pt"
CHAIR_MODEL_WEIGHTS = "yolo26n.pt" 
REID_WEIGHTS = WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"

# ==================================================
# HARDWARE SETTINGS
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

# ==================================================
# AI TUNING PARAMETERS
# ==================================================
CONF_THRESH = 0.4 
IOU_THRESH = 0.5

# ==================================================
# PREPROCESSING SETTINGS
# ==================================================
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

ENABLE_MEDIAN_BLUR = True
BLUR_KERNEL_SIZE = 3