# Retrieval-Augmented Generation (RAG) System over a Code Repository

This project implements a complete pipeline for a Retrieval-Augmented Generation (RAG) system that operates on a code repository. Given a GitHub URL (in this case, [viarotel-org/escrcpy](https://github.com/viarotel-org/escrcpy)), the system clones the repository, indexes its code files using embeddings with FAISS, and enables natural language query answering by returning relevant file locations.

## Features

- **Repository Cloning and File Extraction:** Automatically clones the repository and extracts code files (e.g., Python, JavaScript, HTML, etc.).
- **Index Building with Embeddings and FAISS:** Computes embeddings for each code file using a SentenceTransformer model and indexes them for fast similarity search.
- **Query Processing and Retrieval:** Computes the embedding for a natural language query and performs a nearest-neighbor search to retrieve the top relevant code files.
- **Evaluation Script:** Includes an evaluation module that computes Recall@10 against a provided dataset.
- **Modular Design for Provider Switching:** Easily switch between embedding or LLM providers by updating the relevant modules.
- **Advanced Techniques (Optional):**  
  - **Query Expansion:** Enhance queries using synonyms or paraphrasing.
  - **Reranker:** Re-score top candidates using a cross-encoder model.
  - **LLM-Generated Summaries:** Optionally integrate an LLM to generate natural language summaries of retrieved code segments.
  
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

## Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - `gitpython`
  - `faiss-cpu`
  - `sentence-transformers`
  - `numpy`
  - `scikit-learn`
  - *(Optional)* `openai`, `transformers`, etc. for integrating different LLM/embedding providers

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
## Usage

### Building the Index

To clone the target repository and build the index, run:

```bash
python indexer.py
```

This script will:
- Clone the [viarotel-org/escrcpy](https://github.com/viarotel-org/escrcpy) repository.
- Extract code files based on defined extensions.
- Compute embeddings for each file and build a FAISS index.

### Running an Interactive Query Session

To start an interactive query session where you can input natural language queries:

```bash
python retriever.py
```
You will be prompted to enter a query. The system will return the top retrieved file locations based on the query.

#### Without LLM Summarization
  To run the RAG system **without** LLM summarization (just code retrieval):
  ```python
  python run_query.py --query "Your query here"
  ```

#### With LLM Summarization (OpenAI)
  To enable **LLM summarization** using OpenAI's GPT models for the retrieved code:
  
  1. Ensure that the `OPENAI_API_KEY` is set in your `.env` file.
  2. Run the following command:
     ```python
     python run_query.py --query "How does the server start?" --summarize true --llm_provider openai
     ```
  This will generate an LLM-based summary of the retrieved code snippets.

#### Using Custom Reranker (Optional)
  You can also use a custom reranker (e.g., `bge-reranker-v2-m3` from Hugging Face) to enhance the relevance of the retrieved code. To use the custom reranker, run:
  ```python
  python run_query.py --query "How does the server start?" --reranker custom
  ```
  You can combine the reranker with LLM summarization like this:
  ```python
  python run_query.py --query "How does the server start?" --summarize true --llm_provider openai --reranker custom
  ```

### Evaluation

To evaluate the system using a reference dataset, run the following command:

```bash
python evaluator.py
```
Ensure you have an evaluation dataset file (e.g., evaluation_dataset.json) that contains a list of query objects with their expected file paths. The script will calculate and display the Recall@10 metric based on the retrieved results.
