import torch
from index import embed_query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from summarize import generate_summary
from config import LLM_PROVIDER

reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")

def query_index(query: str, docs, index, model, top_k: int = 10):
    
    query_embedding = embed_query(query, model=model)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append(docs[idx])
    return results

def rerank_results(query, results, top_k=10):
    pairs = [(query, doc["content"]) for doc in results]
    inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)
    scores = scores.numpy()
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in reranked[:top_k]]

def retrieve_and_summarize(query: str, docs, index, model, top_k: int = 10, generate_summaries: bool = True):
    results = query_index(query, docs, index, model, top_k=top_k)
    #results = rerank_results(query, results, top_k) #commented out as the reranker rsults didnt improve recall % much, but took very long
    if generate_summaries:
        summarized_results = []
        for doc in results:
            summary, latency = generate_summary(doc["content"], provider=LLM_PROVIDER)
            summarized_results.append({
                "filename": doc["filename"][5:],
                "summary": summary,
                "summary_latency": latency
            })
        return summarized_results
    else:
        return results
