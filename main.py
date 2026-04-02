"""
Run the Agent
-------------
Choose one of the three options below — text query, image path, or both.
"""

from PIL import Image as PILImage

from state import AgentState
from vision import VisionProcessor
from encoder import TextEncoder
from retrieval import FAISSRetriever, WebSearchRetriever
from synthesis import SynthesisLLM
from graph import build_graph
from utils import pretty_print_recipe


def run(text_query: str = None, img_path: str = None):
    # ── Load all components ──────────────────────────────────────────────────
    vision_processor = VisionProcessor()
    print("VisionProcessor is ready.")

    text_encoder = TextEncoder()
    print("TextEncoder is ready.")

    faiss_retriever = FAISSRetriever()
    web_retriever = WebSearchRetriever()
    print("Retrievers are ready.")

    synthesis_llm = SynthesisLLM()
    print("SynthesisLLM is ready.")

    # Build agent graph
    app = build_graph(vision_processor, text_encoder, faiss_retriever, web_retriever, synthesis_llm)

    # Prepare initial state
    image = PILImage.open(img_path).convert("RGB") if img_path else None

    initial_state: AgentState = {
        "image": image,
        "text_query": text_query,
        "combined_query": None,
        "intent_words": None,
        "image_type": None,
        "vlm_description": None,
        "vlm_confidence": None,
        "query_embedding": None,
        "retrieved_recipes": None,
        "retrieval_source": None,
        "final_recipe": None,
        "messages": [],
        "fallback_count": 0,
        "error": None,
    }

    final_state = app.invoke(initial_state)

    # Handle fallback (agent asked for clarification)
    if final_state.get("image_type") == "unknown" and not final_state.get("final_recipe"):
        last_msg = final_state["messages"][-1]["content"]
        print(f"\n[Agent] {last_msg}")
        user_input = input("Your answer: ").strip()
        final_state["text_query"] = user_input
        final_state["image"] = None
        final_state = app.invoke(final_state)

    recipe = final_state.get("final_recipe")
    if recipe:
        pretty_print_recipe(recipe)
    else:
        print("\n[Agent] Sorry, I couldn't generate a recipe. Please try again.")

    return final_state


def inspect_state(final_state: dict):
    """Optional — useful for debugging intermediate state."""
    print("image_type: ", final_state.get("image_type"))
    print("vlm_description: ", final_state.get("vlm_description"))
    print("vlm_confidence: ", final_state.get("vlm_confidence"))
    print("retrieval_source: ", final_state.get("retrieval_source"))
    print("candidates found: ", len(final_state.get("retrieved_recipes") or []))
    print("Conversation log:")
    for m in final_state.get("messages", []):
        print(f" [{m['role']}] {m['content']}")


if __name__ == "__main__":
    # text_query = "Give me a recipe based on these ingredients"
    text_query = "How to make fried chicken?"
    img_path = "data/images/ingredients5.jpg"  # set to None to skip image input

    final_state = run(text_query=text_query, img_path=img_path)
    inspect_state(final_state)
