"""
Retrieval — FAISS + Web Search Fallback
----------------------------------------
FAISSRetriever: loads the on-disk index and does nearest-neighbour search.
WebSearchRetriever: falls back to Tavily web search when FAISS confidence
is below the similarity threshold.
"""

import os
import json
import numpy as np
import faiss as _faiss
from tavily import TavilyClient

from config import (
    faiss_index_local_path,
    recipes_metdata_local_path,
    top_k,
)


class FAISSRetriever:
    def __init__(self):
        print(f"Loading index from {faiss_index_local_path}...")
        self.index = _faiss.read_index(faiss_index_local_path)
        with open(recipes_metdata_local_path) as f:
            self.metadata = json.load(f)
        print(f"{self.index.ntotal:,} recipes loaded.")
        print("FAISSRetriever is initalized.")

    def search(self, embedding: np.ndarray, k: int = top_k):
        q = embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q, k)
        recipes = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            recipe = self.metadata[idx].copy()
            recipe["similarity_score"] = float(score)
            recipes.append(recipe)

        if len(scores[0]):
            best = float(scores[0][0])
        else:
            best = 0.0

        return recipes, best


class WebSearchRetriever:
    def __init__(self):
        self.client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        print("WebSearchRetriever is initalized.")

    def search(self, query: str, k: int = top_k):
        web_response = self.client.search(
            query=f"recipe for {query}",
            search_depth="advanced",
            max_results=k,
            include_answer=True,
        )

        search_results = []
        for result_item in web_response.get("results", []):
            search_results.append({
                "title": result_item.get("title", "Web result"),
                "ingredients": [],
                "steps": [],
                "url": result_item.get("url", ""),
                "raw_content": result_item.get("content", ""),
                "similarity_score": None,
            })
        return search_results
