"""
LangGraph Agent — Nodes & Graph
---------------------------------
All node functions (vision, fallback, encode_query, retrieval, synthesis)
and the routing function are defined here.
"""

import hashlib
import numpy as np
import spacy
import faiss

from state import AgentState
from config import (
    similarity_threshold,
    max_fallbacks,
    max_index_size,
    faiss_index_local_path,
    recipes_metdata_local_path,
)

# Load a small English model from SpaCy
nlp = spacy.load("en_core_web_sm")


def get_nouns(text):
    doc = nlp(text.lower())
    # Extract only Nouns (NOUN) and Proper Nouns (PROPN)
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def vision_node(state: AgentState, vision_processor) -> AgentState:
    """Identifies and obtains ingredients from the image, updates agent state."""
    image = state.get("image")
    if image is None:
        return {
            **state,
            "image_type": None,
            "vlm_description": state.get("text_query", ""),
            "vlm_confidence": 1.0
        }

    result = vision_processor.classify_image(image)
    return {
        **state,
        "image_type": result["type"],
        "vlm_description": result["description"],
        "vlm_confidence": result["confidence"],
        "messages": state["messages"] + [{
            "role": "system",
            "content": f"'{result['type']}': {result['description']} (conf={result['confidence']:.2f})"
        }],
    }


def fallback_node(state: AgentState) -> AgentState:
    """Falls back when the vision node cannot confidently identify the image."""
    count = state.get("fallback_count", 0) + 1
    msg = ("I couldn't confidently identify your image. "
           "Could you describe what you're cooking or what ingredients you have?")
    return {
        **state,
        "fallback_count": count,
        "image_type": "unknown",
        "vlm_description": None,
        "vlm_confidence": 0.0,
        "messages": state["messages"] + [{"role": "assistant", "content": msg}]
    }


def encode_query_node(state: AgentState, text_encoder) -> AgentState:
    vlm_description = state.get("vlm_description") or ""
    text_query = state.get("text_query") or ""

    noun_list = get_nouns(f"{text_query} {vlm_description}")
    unique_nouns = list(set(noun_list))

    ingredients_formatted = ", ".join(unique_nouns)
    clean_query = f"{text_query}: {ingredients_formatted}"

    embedding = text_encoder.encode(clean_query)

    return {
        **state,
        "query_embedding": embedding.tolist(),
        "combined_query": clean_query,
        "intent_words": unique_nouns
    }


def retrieval_node(state: AgentState, faiss_retriever, web_retriever, text_encoder) -> AgentState:
    import json

    embedding = np.array(state["query_embedding"], dtype="float32")
    recipes, best_score = faiss_retriever.search(embedding)

    # Using vlm_description (always contains actual food words the VLM extracted from the image)
    # for relevance check instead of text_query, because using text_query alone fails for vague
    # queries like "give me a recipe for this".
    intent_words = state.get("intent_words", [])

    def is_relevant(recipe):
        title = recipe.get("title", "").lower()
        # Look in ingredients too to avoid false-negative web searches
        ingredients = " ".join(recipe.get("ingredients", [])).lower()
        return any(word in title or word in ingredients for word in intent_words)

    faiss_relevant = any(is_relevant(r) for r in recipes)

    if best_score >= similarity_threshold and faiss_relevant:
        source = "faiss"
        print(f"FAISS score {best_score:.3f}, candidates relevant")
    else:
        # Fixed the print statement variable name
        reason = "below threshold" if best_score < similarity_threshold else "not relevant"
        print(f"FAISS score {best_score:.3f} {reason} for {intent_words}, doing web search")

        # Use full text query for web search context, fall back to clean query
        search_query = state.get("text_query") or state.get("combined_query")
        recipes = web_retriever.search(search_query)
        source = "web" if recipes else "none"

        # For new web results, ingest web results into FAISS and update
        if source == "web" and recipes:

            # creates hash title + first 100 chars of content as the unique content identifier
            def content_hash(title: str, content: str) -> str:
                return hashlib.md5(f"{title[:50]}:{content[:100]}".encode()).hexdigest()

            # avoid duplicate contents using content hash, not just URL (because URL-only dedup
            # fails when the same recipe appears under a different URL or when the same URL
            # returns slightly different scraped content across queries)
            existing_urls = {r.get("url", "") for r in faiss_retriever.metadata}
            existing_hashes = {
                content_hash(r.get("title", ""), "".join(r.get("steps", [])) if isinstance(r.get("steps"), list) else r.get("raw_content", ""))
                for r in faiss_retriever.metadata
            }

            new_texts = []
            new_meta = []

            for r in recipes:
                url = r.get("url", "")
                chash = content_hash(r.get("title", ""), r.get("raw_content", ""))

                # Skip if duplicate by URL or content hash
                if (url and url in existing_urls) or chash in existing_hashes:
                    print(f" Skipping duplicate: {r.get('title', url)}")
                    continue

                title = r.get("title", "")
                content = r.get("raw_content", "")[:300]
                new_texts.append(f"{title}: {content}")
                new_meta.append({
                    "title": title,
                    "ingredients": r.get("ingredients", []),
                    "steps": r.get("steps", []),
                    "url": url,
                    "raw_content": r.get("raw_content", ""),  # kept for future hash checks
                })

            if new_texts and faiss_retriever.index.ntotal < max_index_size:
                slots_left = max_index_size - faiss_retriever.index.ntotal
                new_texts = new_texts[:slots_left]
                new_meta = new_meta[:slots_left]

                new_embeddings = text_encoder.encode_batch(new_texts)
                faiss_retriever.index.add(new_embeddings)
                faiss_retriever.metadata.extend(new_meta)

                # Persist to disk so knowledge is stored even during session restarts
                faiss.write_index(faiss_retriever.index, faiss_index_local_path)
                with open(recipes_metdata_local_path, "w") as f_out:
                    json.dump(faiss_retriever.metadata, f_out)

                print(f"Added {len(new_texts)} new recipes. Index now has {faiss_retriever.index.ntotal:,} recipes.")

            elif faiss_retriever.index.ntotal >= max_index_size:
                print(f" Index at capacity ({max_index_size:,}), skipping ingestion.")

    return {
        **state,
        "retrieved_recipes": recipes,
        "retrieval_source": source,
        "messages": state["messages"] + [{
            "role": "system",
            "content": f" source={source}, {len(recipes)} candidates, faiss_relevant={faiss_relevant}"
        }]
    }


def synthesis_node(state: AgentState, synthesis_llm) -> AgentState:
    # Always use the combined query so Llama knows BOTH intent and ingredients
    # query  = state.get("combined_query") or state.get("text_query") or state.get("vlm_description", "")
    query = state.get("text_query") or state.get("combined_query")
    recipe = synthesis_llm.synthesise(query, state.get("retrieved_recipes") or [], state.get("image_type"))
    return {
        **state,
        "final_recipe": recipe,
        "messages": state["messages"] + [{
            "role": "assistant",
            "content": f"Recipe ready: {recipe.get('title', '')}"
        }]
    }


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_vision(state: AgentState) -> str:
    """Routing function to decide which node to call next based on agent state."""
    if state.get("image_type") is None:
        return "encode_query"
    if state.get("image_type") in ("ingredients", "dish") and state.get("vlm_confidence", 0) >= 0.5:
        return "encode_query"
    if state.get("fallback_count", 0) >= max_fallbacks:
        return "encode_query"

    return "fallback"
