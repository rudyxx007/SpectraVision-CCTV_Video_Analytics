import torch
from pathlib import Path

# ==================================================
# HARDWARE OPTIMIZATION (TENSOR CORES)
# ==================================================
# FIX: Using "cuda:0" ensures compatibility with both YOLO and StrongSORT
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if "cuda" in DEVICE:
    # Enable TF32 (TensorFloat-32) on Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark 
    torch.backends.cudnn.benchmark = True
    
    # Floating Point 16 (Half Precision)
    USE_FP16 = True
else:
    USE_FP16 = False

# ==================================================
# PATHS AND DIRECTORIES
# ==================================================
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
PROJECT_ROOT = CODE_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

INPUT_VIDEO_DIR = DATA_DIR / "input_video"
OUTPUT_LOGS_DIR = DATA_DIR / "logs"
MASK_DIR = DATA_DIR / "masks"

VIDEO_SOURCE = INPUT_VIDEO_DIR / "cctv_video.mp4"
ROI_MASK_PATH = MASK_DIR / "room_mask.png" 

# ==================================================
# MODEL WEIGHTS
# ==================================================
POSE_MODEL_WEIGHTS = WEIGHTS_DIR / "yolo26s-pose.pt"
CHAIR_MODEL_WEIGHTS = "yolo26n.pt" 
REID_WEIGHTS = WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"

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