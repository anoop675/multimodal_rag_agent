"""
Vision Module — CLIP-ViT + Qwen2.5-3B VLM
------------------------------------------
CLIP-ViT encodes the image into a 512-dim embedding.
Qwen2.5-3B classifies it as Type A (ingredients) or Type B (dish)
and extracts a text description.
"""

import re
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from config import clip_model_id, vlm_model_id, device

vlm_system_prompt = """You are a food identification assistant. Be specific and detailed.

Look at this image and answer:
1. TYPE: Is this (A) raw/uncooked ingredients laid out, or (B) a completed cooked dish?
   If you cannot tell, answer UNKNOWN.
2. DESCRIPTION: List every ingredient or food item you can see, separated by commas.
   Be specific — name the pasta shape, type of vegetable, type of cheese etc.
3. CONFIDENCE: A number from 0.0 to 1.0.

Reply in exactly this format (no extra text):
TYPE: A
DESCRIPTION: rigatoni pasta, canned crushed tomatoes, olive oil, parmesan cheese, white onion, garlic cloves, fresh basil, dried oregano, salt, black pepper
CONFIDENCE: 0.95
"""


class VisionProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Loading on {self.device}...")

        self.clip_model = CLIPModel.from_pretrained(clip_model_id)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        # self.clip_model.requires_grad_(False)

        self.clip_proc = CLIPProcessor.from_pretrained(clip_model_id)

        # On Apple Silicon, avoid device_map="auto" which splits across CPU+MPS
        # and causes oversized attention buffers. Load in float32 on CPU instead.
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            vlm_device = "cpu"  # keep VLM on CPU to avoid MPS buffer size limits
            vlm_dtype = torch.float32
        elif self.device == "cuda":
            vlm_device = "auto"
            vlm_dtype = torch.float16
        else:
            vlm_device = "cpu"
            vlm_dtype = torch.float32

        self.vlm_model = AutoModelForImageTextToText.from_pretrained(
            vlm_model_id,
            dtype=vlm_dtype,
            device_map=vlm_device,
            ignore_mismatched_sizes=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
        print("VisionProcessor is initialized.")

    def encode_image(self, image: PILImage.Image) -> np.ndarray:
        inputs = self.clip_proc(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().astype("float32")
        embedding = embedding.flatten()
        return embedding

    def classify_image(self, image: PILImage.Image) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": vlm_system_prompt},
            ],
        }]
        text_in = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, _ = process_vision_info(messages)
        vlm_device = next(self.vlm_model.parameters()).device
        inputs = self.vlm_processor(text=[text_in], images=img_inputs,
                                    return_tensors="pt").to(vlm_device)
        with torch.no_grad():
            out_ids = self.vlm_model.generate(**inputs, max_new_tokens=150)
        trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
        raw = self.vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed_raw = self.parse_raw(raw)
        return parsed_raw

    def parse_raw(self, raw: str) -> dict:
        return self.parse(raw)

    def parse(self, text: str) -> dict:
        extracted_recipe_info = {
            "type": "unknown",
            "description": "",
            "confidence": 0.0
        }
        type_match = re.search(r"TYPE:\s*([AB]|UNKNOWN)", text, re.IGNORECASE)
        description = re.search(r"DESCRIPTION:\s*(.+)", text)
        conf = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if type_match:
            t = type_match.group(1).upper()
            extracted_recipe_info["type"] = "ingredients" if t == "A" else "dish" if t == "B" else "unknown"
        if description:
            extracted_recipe_info["description"] = description.group(1).strip()
        if conf:
            extracted_recipe_info["confidence"] = float(conf.group(1))

        return extracted_recipe_info
