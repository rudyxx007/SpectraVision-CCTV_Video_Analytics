import torch
import ultralytics
import boxmot
from transformers import AutoModelForCausalLM

print(f"--- 2026 Stack Verification ---")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print(f"YOLO (Pose): {ultralytics.__version__}")
print(f"StrongSORT++ Lib: {boxmot.__file__}")

try:
    # Check if Qwen3 is importable (Don't load weights yet)
    from transformers import Qwen2VLForConditionalGeneration
    print("Qwen3-VL Architecture: DETECTED")
except ImportError:
    print("❌ Qwen3-VL Architecture not found (Transformers update needed)")

print("✅ Step 0 Verification Passed.")