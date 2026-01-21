import torch
import os
from pathlib import Path
from boxmot import StrongSort

# Force-fix the environment variable issue locally
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def download_reid_weights():
    # 1. Get the directory where THIS script is located (D:/.../Code)
    current_script_dir = Path(__file__).resolve().parent
    
    # 2. Go one level up to the Project Root and select the weights folder
    # This is equivalent to "../weights"
    weights_dir = current_script_dir.parent / "weights"
    
    # 3. Create the folder if it doesn't exist
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Define the final path for the weights
    reid_path = weights_dir / "osnet_x0_25_msmt17.pt"
    
    print(f"📂 Project Root: {current_script_dir.parent}")
    print(f"📥 Targeting weights at: {reid_path}")

    try:
        # 5. Initialize StrongSort to download the weights
        tracker = StrongSort(
            reid_weights=reid_path, 
            device=torch.device(0),
            half=True 
        )
        print(f"✅ Success! Re-ID Weights are at: {reid_path}")
    except Exception as e:
        print(f"❌ Error during setup: {e}")

if __name__ == "__main__":
    download_reid_weights()