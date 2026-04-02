"""
Build FAISS Index
-----------------
Downloads RecipeNLG from HuggingFace, encodes each recipe with CLIP,
and saves a FAISS index to disk. (one-time offline step)

Skip this script if 'data/recipes.index' already exists.
"""

import os
import json
import numpy as np
import faiss
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm

from config import (
    clip_model_id,
    recipe_dataset_path,
    faiss_index_local_path,
    recipes_metdata_local_path,
    dataset_limit,
    batch_size,
    device,
)


def encode_texts(texts, clip_model, clip_processor):
    inputs = clip_processor(text=texts, return_tensors="pt", truncation=True,
                            max_length=77, padding=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
    # get_text_features returns a tensor directly, but if it returns an object, extract it
    if hasattr(embedding, "pooler_output"):
        embedding = embedding.pooler_output
    elif hasattr(embedding, "last_hidden_state"):
        embedding = embedding.last_hidden_state[:, 0, :]

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.cpu().numpy().astype("float32")
    return embedding


def build_index():
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_model.to(device)
    clip_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    print("Loading dataset...")
    ds = load_dataset(recipe_dataset_path, split="train", streaming=True)

    index = faiss.IndexFlatIP(512)  # for exact search (increase in index size causes higher search latency)
    # index = faiss.IndexHNSWFlat(512, 32)  # 32 = graph connections, for approximate search if larger index (more accurate, but more memory)
    # index.hnsw.efConstruction = 200
    # index.hnsw.efSearch = 128
    metadata = []
    batch_t = []
    batch_m = []
    count = 0

    for recipe in tqdm(ds, total=dataset_limit, unit="recipe"):
        if count >= dataset_limit:
            break

        title = recipe.get("title") or ""
        ingredients = recipe.get("ingredients") or ""
        steps = recipe.get("directions") or recipe.get("description") or ""
        url = recipe.get("url") or ""

        # ingredients is a plain string on this dataset, not a list
        if isinstance(ingredients, str):
            ingredients = [i.strip() for i in ingredients.split("\n") if i.strip()]

        # steps is also a plain string on this dataset
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.split("\n") if s.strip()]

        ingredients_str = ", ".join(ingredients[:10])
        batch_t.append(f"{title}: {ingredients_str}")
        batch_m.append({
            "title": title,
            "ingredients": ingredients,
            "steps": steps,
            "url": url,
        })
        count += 1

        if len(batch_t) == batch_size:
            index.add(encode_texts(batch_t, clip_model, clip_processor))
            metadata.extend(batch_m)
            batch_t, batch_m = [], []

    if batch_t:
        index.add(encode_texts(batch_t, clip_model, clip_processor))
        metadata.extend(batch_m)

    faiss.write_index(index, faiss_index_local_path)
    with open(recipes_metdata_local_path, "w") as f:
        json.dump(metadata, f)

    print(f"\nDone! {index.ntotal:,} recipes indexed.")


if __name__ == "__main__":
    build_index()
