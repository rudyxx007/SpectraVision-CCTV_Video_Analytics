import torch
from pathlib import Path

# ==================================================
# HARDWARE
# ==================================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_FP16 = True if "cuda" in DEVICE else False

# ==================================================
# DIRECTORIES
# ==================================================
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
PROJECT_ROOT = CODE_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Input/Output
VIDEO_SOURCE = DATA_DIR / "input_video" / "office_cctv.mp4"
ROI_MASK_PATH = DATA_DIR / "masks" / "room_mask.png" 
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True) 

# ==================================================
# PREPROCESSING
# ==================================================
# Preprocessing Switches
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
ENABLE_MEDIAN_BLUR = True
BLUR_KERNEL_SIZE = 3

# ==================================================
# MODEL WEIGHTS
# ==================================================
# Phase 1: Perception
POSE_MODEL_WEIGHTS = WEIGHTS_DIR / "yolo26s-pose.pt"
CHAIR_MODEL_WEIGHTS = "yolo26n.pt" 
REID_WEIGHTS = WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"

# Phase 2: Verification
SAM_WEIGHTS = WEIGHTS_DIR / "sam3.pt"  # Your downloaded SAM 3 weights
VLM_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct" # HuggingFace Qwen3-VL ID

# ==================================================
# THRESHOLDS
# ==================================================
CONF_THRESH = 0.25   
IOU_THRESH = 0.5     

# Logic Thresholds
SITTING_ANGLE_THRESH = 110
SAM_TRIGGER_IOU = 0.10     
SGG_INTERVAL = 5.0         

# Dashboard
KPI_FILE = LOGS_DIR / "kpi_stats.csv"
LIVE_STATE_FILE = LOGS_DIR / "live_state.json"