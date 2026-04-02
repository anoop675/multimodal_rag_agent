from typing import TypedDict, Literal, Optional
from PIL import Image as PILImage


class AgentState(TypedDict):  # Agent state stored in RAM for state context
    # For multimodal input
    image: Optional[PILImage.Image]
    text_query: Optional[str]
    combined_query: Optional[str]  # used for merging intent + ingredients -> VLM
    intent_words: Optional[str]

    image_type: Optional[Literal["ingredients", "dish", "unknown"]]  # either image is a completed dish or ingredients
    vlm_description: Optional[str]
    vlm_confidence: Optional[float]

    # For retrieval
    query_embedding: Optional[list]
    retrieved_recipes: Optional[list]
    retrieval_source: Optional[Literal["faiss", "web", "none"]]

    final_recipe: Optional[dict]  # for final recipe

    # For general dialogue
    messages: list
    fallback_count: int
    error: Optional[str]
