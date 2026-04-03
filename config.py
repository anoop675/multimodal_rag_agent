import os
import torch
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Model IDs
vlm_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
clip_model_id = "openai/clip-vit-base-patch32"
synthesis_model_id = "llama-3.3-70b-versatile"

# RAG settings
top_k = 5
similarity_threshold = 0.50
max_fallbacks = 2

# Paths & Data
recipe_dataset_path = "corbt/all-recipes"
os.makedirs("data", exist_ok=True)
faiss_index_local_path = "data/recipes.index"
dataset_limit = 1000000
batch_size = 256
recipes_metdata_local_path = "data/recipes_metadata.json"
max_index_size = 1000500

# Computing Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
