import cv2
import re
from PIL import Image
import numpy as np
import torch

import config

class VLMDetector:
    """Unified VLM Detector utilizing nvidia/LocateAnything-3B via native Transformers and bitsandbytes."""
    def __init__(self):
        print(f"[INFO] VLM Detector: Initializing LocateAnything-3B via Transformers (8-bit Quantized)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            model_path = "nvidia/LocateAnything-3B"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True, # Compress 6GB model to 3GB for RTX 5050
                trust_remote_code=True,
            ).eval()
            print("[SUCCESS] Transformers Multimodal Model Initialized.")
        except Exception as e:
            print(f"[WARNING] Failed to load Transformers model: {e}")
            self.model = None

    def compute_pas_score(self, response_text: str) -> float:
        if "<box>" not in response_text:
            return 0.0
        return 1.0

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str) -> dict:
        if self.model is None:
            return {"answer": ""}

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]
        
        try:
            text = self.processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=images, videos=videos, return_tensors="pt"
            ).to(self.device)

            # Ensure pixel values are half precision if model is quantized
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            response = self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                use_cache=True,
                generation_mode="hybrid",
                temperature=0.2, # Low temp for accurate boxes
                do_sample=True,
                verbose=False,
            )

            answer = response[0] if isinstance(response, tuple) else response
            
            # Apply PAS Hallucination Filter
            pas_score = self.compute_pas_score(answer)
            if pas_score < 0.5:
                print("[WARNING] PAS Filter blocked a hallucination.")
                answer = ""
                
            return {"answer": answer}
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return {"answer": ""}

    def detect_chairs(self, frame_bgr):
        if self.model is None:
            return [], []

        pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        w, h = pil_image.size

        # PREDICATE DICTIONARY
        occupied_predicates = [
            # Workspace Occupied (Including Occlusions)
            "person sitting on a chair",
            "person slouching on an office chair",
            "person working at a desk",
            "partially occluded person sitting down",
            "person sitting at a desk partially hidden behind a monitor",
            "person sitting partially visible",
            "person's head or shoulders visible sitting at a desk",
            
            # General People (Walking/Standing, including occlusions)
            "person walking",
            "person standing",
            "person standing next to a desk",
            "person crouching or bending over",
            "partially occluded person walking",
            "person partially hidden by a wall or object"
        ]
        empty_predicates = [
            # Workspace Available (Including occlusions/distractors)
            "empty office chair",
            "unoccupied chair",
            "chair with no one sitting on it",
            "chair with a jacket or bag on it but no person",
            "partially hidden empty chair behind a desk",
            "empty workspace desk with an unoccupied chair",
            "chair tucked under a desk"
        ]

        # Query 1: Occupied Chairs
        occ_query = "</c>".join(occupied_predicates)
        res_occ = self.predict(pil_image, f"Locate all the instances that match the following description: {occ_query}.")
        occ_parsed = self.parse_boxes(res_occ.get("answer", ""), w, h)
        
        # Query 2: Empty Chairs
        emp_query = "</c>".join(empty_predicates)
        res_empty = self.predict(pil_image, f"Locate all the instances that match the following description: {emp_query}.")
        empty_parsed = self.parse_boxes(res_empty.get("answer", ""), w, h)

        # Formatting for OC-SORT Tracker: [x1, y1, x2, y2, conf, cls]
        occupied_boxes = [[b['x1'], b['y1'], b['x2'], b['y2'], 1.0, 56] for b in occ_parsed]
        empty_boxes = [[b['x1'], b['y1'], b['x2'], b['y2'], 1.0, 56] for b in empty_parsed]

        return occupied_boxes, empty_boxes

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list:
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000.0 * image_width,
                "y1": y1 / 1000.0 * image_height,
                "x2": x2 / 1000.0 * image_width,
                "y2": y2 / 1000.0 * image_height,
            })
        return boxes
