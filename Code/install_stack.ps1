# Jio CCTV Analytics - "Dream Team" Installer

Write-Host "Starting Install for RTX 5050 Stack..." -ForegroundColor Green

# 1. Core Utilities
pip install opencv-python supervision gitpython pyyaml tqdm scipy ninja
pip install "numpy>=2.0,<2.3.0"  # Locking Numpy for OpenCV stability

# 2. YOLOv12-Pose (via Ultralytics)
Write-Host "Installing YOLOv12-Pose..." -ForegroundColor Cyan
pip install ultralytics

# 3. StrongSORT++ (The Tracker)
# We use 'boxmot' because it contains the Official StrongSORT++ implementation
# and compiles correctly on Windows 11/CUDA 13.
Write-Host "Installing StrongSORT++..." -ForegroundColor Cyan
pip install boxmot lapx

# 4. Qwen3-VL-8B (The SGG VLM)
# We need the latest transformers and quantization tools
Write-Host "Installing Qwen3-VL Dependencies..." -ForegroundColor Cyan
pip install git+https://github.com/huggingface/transformers.git@main
pip install accelerate bitsandbytes qwen_vl_utils
pip install flash-attn --no-build-isolation

# 5. SAM 3 (Segmentation)
Write-Host "Installing SAM 3..." -ForegroundColor Cyan
# Clone and install SAM 3 manually to ensure it doesn't downgrade Numpy
if (!(Test-Path "SAM3")) {
    git clone https://github.com/facebookresearch/sam3.git
}
pip install -e SAM3 --no-deps

Write-Host "✅ Step 0 Complete. Environment Ready." -ForegroundColor Green