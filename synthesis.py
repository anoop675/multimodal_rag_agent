"""
Synthesis — LLM Recipe Generator
----------------------------------
Uses Groq (Llama-3) to synthesise a structured recipe JSON from the
retrieved candidates and the original query.
"""

import os
import re
import json as _json
from groq import Groq as _Groq

from config import synthesis_model_id

synthesis_llm_system_prompt = """You are a professional recipe writer and chef.
Given a user query and retrieved recipe candidates, synthesise one perfect structured recipe.
If the user provides a list of ingredients, create a recipe that uses THOSE SPECIFIC ingredients.
If candidates are not relevant, use your own knowledge to create the best recipe.

**STRICT VISUAL CONSTRAINT:** Make a recipe using the VLM identified ingredients [vlm_description].

First, think step by step inside <think>...</think> tags:
- Which candidate best matches the query and why?
- Are the ingredients consistent with the dish the user asked for?
- What adjustments should be made?

Then output ONLY valid JSON after the closing </think> tag, matching this schema:
{
  "title": "...",
  "description": "One sentence.",
  "ingredients": [{ "item": "...", "quantity": "..." }],
  "steps": [{ "step": 1, "instruction": "..." }],
  "tips": ["..."],
}"""


class SynthesisLLM:
    def __init__(self):
        self.client = _Groq(api_key=os.environ["GROQ_API_KEY"])
        print("SynthesisLLM is initialized.")

    def synthesise(self, query: str, candidates: list, image_type=None) -> dict:
        context = ""
        for i, r in enumerate(candidates[:5], 1):
            context += f"\n--- Candidate {i}: {r.get('title', 'Untitled')} ---\n"
            if r.get("raw_content"):
                context += r["raw_content"][:800] + "\n"
            else:
                if r.get("ingredients"):
                    context += "Ingredients: " + ", ".join(str(x) for x in r["ingredients"][:20]) + "\n"
                if r.get("steps"):
                    steps = r["steps"] if isinstance(r["steps"], list) else [r["steps"]]
                    context += "Steps: " + " ".join(str(s) for s in steps[:5]) + "\n"

        type_hint = ""
        if image_type == "ingredients":
            type_hint = "The user has raw ingredients and wants a recipe using them."
        elif image_type == "dish":
            type_hint = "The user has a photo of a completed dish and wants to recreate it."

        user_msg = f"User query: {query}\n{type_hint}\n\nCandidates:{context}\n\nSynthesise the best recipe."

        response = self.client.chat.completions.create(
            model=synthesis_model_id,
            messages=[
                {"role": "system", "content": synthesis_llm_system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )

        raw = response.choices[0].message.content

        # print chain of thought if present
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            print("\n── LLM Chain of Thought ──────────────────────────────")
            print(think_match.group(1).strip())
            print("──────────────────────────────────────────────────────\n")

        # Strip think tags and parse JSON
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _json.loads(raw)
