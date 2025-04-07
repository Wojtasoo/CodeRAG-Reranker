import os
import glob
import git
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import faiss
import tiktoken
from config import EMBEDDING_MODEL, EMBEDDING_PROVIDER

client = OpenAI()

def clone_repository(repo_url: str, clone_dir: str = "repo"):
    if not os.path.exists(clone_dir):
        print(f"Cloning repository from {repo_url} into {clone_dir}")
        git.Repo.clone_from(repo_url, clone_dir)
    else:
        print(f"Repository already cloned in {clone_dir}")
    return clone_dir

def extract_files(repo_dir: str, extensions: list = None):
    return [f for f in glob.glob(os.path.join(repo_dir, "**"), recursive=True) if os.path.isfile(f)]

def load_documents(file_paths: list):
    docs = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"filename": fp, "content": content})
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    return docs

def chunk_text(text, max_tokens=8192, model_name="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model_name)
    
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def build_embeddings(docs, model_name: str = EMBEDDING_MODEL):
    if EMBEDDING_PROVIDER == "sentence_transformers":
        model = SentenceTransformer(model_name)
        texts = [doc["content"] for doc in docs]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        new_docs = docs
    elif EMBEDDING_PROVIDER == "openai":
        texts = []
        new_docs = []
        for doc in docs:
            chunks = chunk_text(doc["content"], max_tokens=8192, model_name="text-embedding-ada-002")
            for chunk in chunks:
                texts.append(chunk)
                new_docs.append({
                    "filename": doc["filename"],
                    "content": chunk
                })
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = client.embeddings.create(input=batch_texts, model="text-embedding-ada-002")
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        embeddings = np.array(all_embeddings)
        model = None
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
    return new_docs, embeddings, model

def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def build_index(repo_url: str):
    repo_dir = clone_repository(repo_url)
    file_paths = extract_files(repo_dir)
    docs = load_documents(file_paths)
    docs, embeddings, model = build_embeddings(docs)
    index = build_faiss_index(embeddings)
    return docs, embeddings, index, model

def embed_query(query: str, embedding_provider=EMBEDDING_PROVIDER, model=None):
    if embedding_provider == "sentence_transformers":
        query_embedding = model.encode([query], convert_to_numpy=True)
    elif embedding_provider == "openai":
        response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
        query_embedding = np.array(response.data[0].embedding) # Convert to numpy array
    else:
        raise ValueError(f"Unknown embedding provider: {embedding_provider}")

    return query_embedding
