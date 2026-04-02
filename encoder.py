"""
CLIP Text Encoder
-----------------
Encodes the VLM description (or raw text query) into the same
512-dim space as the recipe index.
"""

import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

from config import clip_model_id


class TextEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_id)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        print("TextEncoder is initialized.")

    def encode(self, text: str) -> np.ndarray:
        inputs = self.clip_processor(text=[text], return_tensors="pt",
                                     truncation=True, max_length=77, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        # get_text_features returns a tensor directly, but if it returns an object, extract it
        if hasattr(embedding, "pooler_output"):
            embedding = embedding.pooler_output
        elif hasattr(embedding, "last_hidden_state"):
            embedding = embedding.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().astype("float32")
        embedding = embedding.flatten()

        return embedding

    def encode_batch(self, texts: list) -> np.ndarray:
        inputs = self.clip_processor(text=texts, return_tensors="pt",
                                     truncation=True, max_length=77, padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        if hasattr(embedding, "pooler_output"):
            embedding = embedding.pooler_output
        elif hasattr(embedding, "last_hidden_state"):
            embedding = embedding.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().astype("float32")
        return embedding
