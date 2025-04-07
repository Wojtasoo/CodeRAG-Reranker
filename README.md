# Retrieval-Augmented Generation (RAG) System over a Code Repository

This project implements a complete pipeline for a Retrieval-Augmented Generation (RAG) system that operates on a code repository. Given a GitHub URL (in this case, [viarotel-org/escrcpy](https://github.com/viarotel-org/escrcpy)), the system clones the repository, indexes its code files using embeddings with FAISS, and enables natural language query answering by returning relevant file locations.

## Features

- **Repository Cloning and File Extraction:** Automatically clones the repository and extracts code files (e.g., Python, JavaScript, HTML, etc.).
- **Index Building with Embeddings and FAISS:** Computes embeddings for each code file using a SentenceTransformer model and indexes them for fast similarity search.
- **Query Processing and Retrieval:** Computes the embedding for a natural language query and performs a nearest-neighbor search to retrieve the top relevant code files.
- **Evaluation Script:** Includes an evaluation module that computes Recall@10 against a provided dataset.
- **Modular Design for Provider Switching:** Easily switch between embedding or LLM providers by updating the relevant modules.
- **Advanced Techniques:**  
  - **Query Expansion:** Enhanced queries using synonyms or paraphrasing with AI.
  - **Reranker:** Re-score top candidates using a bge-reranker-v2-m3 model.
  - **LLM-Generated Summaries:** Integrate an LLM to generate natural language summaries of retrieved code segments.
  
## Project Structure
```bash
rag_system/
├── README.md              # Project documentation and usage guide
├── requirements.txt       # Python dependencies
├── main.py                # Optional entry point to run full pipeline
├── index.py             # Handles repo cloning, file extraction, embedding, and indexing
├── retrieve.py           # Handles query processing and retrieval
├── evaluate.py           # Evaluation script to compute Recall@10
```

## Tech Stack

- **Python 3.8+**
- **FAISS** – For fast vector similarity search
- **SentenceTransformers** – For generating embeddings of code/documentation
- **Transformers (HuggingFace)** – For summarization and language model tasks
- **GitPython** – For cloning and managing GitHub repositories
- **JSON** – For storing evaluation datasets and summaries

## Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - `gitpython`
  - `faiss-cpu`
  - `sentence-transformers`
  - `numpy`
  - `scikit-learn`
  - `torch`
  - `openai`
  - `accelerate`
  - `transformers`
  - `tiktoken`
## Installation

1. **Clone the repository for this project:**

   ```bash
   git clone https://github.com/your-username/rag_system.git
   cd rag_system
   ```
2. **Create the Virtual Environment:**

   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your `.env` file (if using OpenAI summarization):**
   Create a `.env` file in the root directory and add the following line:
   ```arduino
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Program

### Configuration with `config.py`

The `config.py` file serves as a central place to manage system-wide settings related to LLM and embedding behavior. It allows you to easily switch between local or remote language models, control embedding sources, and specify model names.

#### Key Options

```python
# LLM provider options: "local", "openai", "custom"
LLM_PROVIDER = "local"

# Embedding provider options: "sentence_transformers" or "openai"
EMBEDDING_PROVIDER = "sentence_transformers"

# Default embedding model when using SentenceTransformer
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```
- **LLM_PROVIDER**  
  Controls which language model provider is used for summarization and other generative tasks.

  **Options:**
  - `"local"` – Uses a local transformer model via `HuggingFace`
  - `"openai"` – Connects to OpenAI's model `gpt 4o-mini` (requires an API key)
  - `"custom"` – Allows integration with a chosen model from HuggingFace `(default:t5-base)`

- **EMBEDDING_PROVIDER**  
  Determines how embeddings are generated for code files and queries.

  **Options:**
  - `"sentence_transformers"` – Uses a local model such as `all-MiniLM-L6-v2` via the `sentence-transformers` library
  - `"openai"` – Uses OpenAI's embedding API (ensure your API key is configured)


To start the system, run the following command from the root directory:
```bash
python main.py
```

This script will:
- Clone the [viarotel-org/escrcpy](https://github.com/viarotel-org/escrcpy) repository.
- Extract files.
- Compute embeddings for each file and build a FAISS index.

### Running an Interactive Query Session

You will be prompted to choose between two modes of operation:

### 1. Interactive Mode (`i`)
- In this mode, you can type natural language queries related to the codebase.
- The system retrieves the top matching code files and generates summaries for each.
- Example: `Enter your query: Where is the websocket server implemented?`

### 2. Evaluation Mode (`e`)
- This mode runs an automated evaluation using a pre-existing dataset of queries and expected answers (`escrcpy-commits-generated.json`).
- The script will calculate and display the Recall@10 metric based on the retrieved results.

> If other button is pressed, the system will default to **Interactive Mode**.
