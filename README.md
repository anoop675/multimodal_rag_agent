# Multi-Modal RAG Recipe Generation Agent


## Setup in PyCharm

### 1. Open the project
File → Open → select the `cook_rag_agent/` folder

### 2. Create a virtual environment
PyCharm will prompt you — choose **New environment using Virtualenv**.
Or manually: `python -m venv .venv`

### 3. Set the interpreter
Settings → Project → Python Interpreter → Add → select `.venv/bin/python`

### 4. Install dependencies
Open the PyCharm terminal and run:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Configure API keys
Edit `.env`:
```
GROQ_API_KEY=your_groq_key      # free at console.groq.com
TAVILY_API_KEY=your_tavily_key  # free at tavily.com
```

### 6. Set run configuration
Run → Edit Configurations → Add → Python
- Script: `main.py`
- Parameters: `--image photos/dish.jpg --query "How to make this?"`
- Working directory: project root

### 7. Build the FAISS index (one-time)
```bash
# Quick smoke test (1000 recipes, ~30 sec)
python scripts/build_index.py --limit 1000

# Full production index (1M recipes, 1-6 hours)
python scripts/build_index.py
```

### 8. Run
```bash
python main.py --image photos/ingredients.jpg --query "How to make fried chicken?"
python main.py --query "pasta with mushrooms and cream"
python main.py --image photos/dish.jpg
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all settings |
| `agent/state.py` | AgentState schema shared across all nodes |
| `agent/graph.py` | LangGraph wiring, all node functions, routing |
| `vision/vlm_processor.py` | Image → type + description (CLIP + Qwen VLM) |
| `vision/text_encoder.py` | Text → CLIP embedding (encode / encode_batch) |
| `retrieval/retriever.py` | FAISS search, web search, content_hash util |
| `synthesis/synthesiser.py` | Llama 3.3 recipe synthesis with chain of thought |
| `scripts/build_index.py` | Offline job: dataset → FAISS index |
| `main.py` | CLI entry point + pretty_print_recipe |

