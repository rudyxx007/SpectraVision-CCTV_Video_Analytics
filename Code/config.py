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
# Real-world indoor surveillance test video for VLM benchmarking
VIDEO_SOURCE = DATA_DIR / "input_video" / "intel_test_video.mp4"
ROI_MASK_PATH = DATA_DIR / "masks" / "room_mask.png" 
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True) 

# ==================================================
# PREPROCESSING
# ==================================================
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
ENABLE_MEDIAN_BLUR = True
BLUR_KERNEL_SIZE = 3

# ==================================================
# UNIFIED VISION-LANGUAGE MODEL (LocateAnything)
# ==================================================
VLM_MODEL_ID = "nvidia/LocateAnything-3B"
VLM_INTERVAL = 2.0  # Run VLM every 2 seconds for ground truth updates

# ==================================================
# TRACKING & KALMAN FILTER (StrongSORT)
# ==================================================
REID_WEIGHTS = WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"
CONF_THRESH = 0.30
IOU_THRESH = 0.5

# ==================================================
# DASHBOARD LOGGING
# ==================================================
KPI_FILE = LOGS_DIR / "kpi_stats.csv"
LIVE_STATE_FILE = LOGS_DIR / "live_state.json"