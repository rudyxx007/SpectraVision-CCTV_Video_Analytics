import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import config

class SceneGraphEngine:
    def __init__(self):
        print("[INFO] SGG: Initializing Qwen2-VL (Semantic Reasoning)...")
        try:
            # Load Model in 4-bit or FP16 to save VRAM
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.VLM_MODEL_ID, 
                torch_dtype=torch.float16,
                device_map=config.DEVICE
            )
            self.processor = AutoProcessor.from_pretrained(config.VLM_MODEL_ID)
            print("[SUCCESS] VLM Initialized.")
        except Exception as e:
            print(f"[WARNING] VLM Init Failed (Running in logic-only mode): {e}")
            self.model = None

    def verify_interaction(self, frame_rgb, person_box, chair_box):
        """
        Crops the interaction zone and asks VLM.
        """
        if self.model is None: return 0.5 # Neutral score if VLM missing

        # 1. Create Crop (Union of boxes)
        x1 = int(min(person_box[0], chair_box[0]))
        y1 = int(min(person_box[1], chair_box[1]))
        x2 = int(max(person_box[2], chair_box[2]))
        y2 = int(max(person_box[3], chair_box[3]))
        
        # Padding
        h, w, _ = frame_rgb.shape
        x1, y1 = max(0, x1-20), max(0, y1-20)
        x2, y2 = min(w, x2+20), min(h, y2+20)

        crop = frame_rgb[y1:y2, x1:x2]
        pil_image = Image.fromarray(crop)

        # 2. Prompt
        prompt = "Is the person sitting on the chair? Answer Yes or No."
        
        # 3. Inference
        inputs = self.processor(
            text=[prompt], images=pil_image, padding=True, return_tensors="pt"
        ).to(config.DEVICE)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=10)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # 4. Parse
        if "yes" in output_text.lower():
            return 0.95
        elif "no" in output_text.lower():
            return 0.05
        else:
            return 0.5