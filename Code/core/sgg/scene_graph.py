import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import cv2
import config

class SceneGraphEngine:
    def __init__(self):
        print(f"[INFO] SGG: Initializing Qwen3-VL ({config.VLM_MODEL_ID})...")
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.VLM_MODEL_ID, 
                torch_dtype=torch.float16 if config.USE_FP16 else torch.float32,
                device_map=config.DEVICE
            )
            self.processor = AutoProcessor.from_pretrained(config.VLM_MODEL_ID)
            print("[SUCCESS] Qwen3-VL Initialized.")
        except Exception as e:
            print(f"[WARNING] Qwen3-VL Init Failed: {e}")
            self.model = None

    def verify_interaction(self, frame_bgr, person_box, chair_box):
        if self.model is None: return 0.5 

        # 1. Create Crop (Union of boxes + padding)
        x1 = int(max(0, min(person_box[0], chair_box[0]) - 20))
        y1 = int(max(0, min(person_box[1], chair_box[1]) - 20))
        x2 = int(min(frame_bgr.shape[1], max(person_box[2], chair_box[2]) + 20))
        y2 = int(min(frame_bgr.shape[0], max(person_box[3], chair_box[3]) + 20))
        
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0: return 0.5

        # Convert OpenCV BGR to PIL RGB
        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # 2. Qwen3-VL Prompt Format
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Is the person sitting on the chair? Answer strictly Yes or No."}
            ]}
        ]
        
        try:
            # 3. Inference
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text], images=[pil_image], padding=True, return_tensors="pt"
            ).to(config.DEVICE)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=5)
                
            output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            # 4. Parse
            if "yes" in output_text.lower(): return 0.95
            elif "no" in output_text.lower(): return 0.05
            else: return 0.5
        except Exception as e:
            print(f"[WARNING] Qwen3-VL Inference Error: {e}")
            return 0.5