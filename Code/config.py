import os
import torch

# --- RTX 50-Series Compatibility Patch ---
try:
    import rtx50_compat
    print("✅ RTX 50-Series compatibility patch applied.")
except ImportError:
    print("⚠️ rtx50-compat not found. Ensure PyTorch 2.9+ is installed.")

# --- Tensor Core Optimizations ---
# This allows the RTX 5050 to use TF32 (TensorFloat-32) on its Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Speed up convolutions by finding the best algorithm for your hardware
torch.backends.cudnn.benchmark = True 

# --- Hardware Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data/input_video/Video 2.mp4")
OUTPUT_CSV = os.path.join(BASE_DIR, "data/logs/occupancy_log.csv")

# --- Model Settings ---
SAM_PROMPT = ["office chair", "person"]
PREDICATES = ["sitting_on", "leaning_on", "standing_near", "occluding"]

print(f"🚀 Environment Ready. GPU: {torch.cuda.get_device_name(0)}")