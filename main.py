from index import build_index
from retrieve import retrieve_and_summarize
from evaluate import evaluate

def run_interactive(docs, embeddings, index, model):
    query = input("Enter your query: ")
    results = retrieve_and_summarize(query, docs, index, model, top_k=10, generate_summaries=True)
    print("\nTop retrieved files with summaries:")
    for res in results:
        print(f"- {res['filename']}\n  Summary: {res['summary']}\n  Summarization Latency: {res['summary_latency']:.2f} sec\n")

def run_evaluation(docs, embeddings, index, model):
    eval_dataset_path = "escrcpy-commits-generated.json"
    print("\nEvaluating retrieval performance:")
    evaluate(eval_dataset_path, docs, index, model)

def main():
    repo_url = "https://github.com/viarotel-org/escrcpy"
    
    docs, embeddings, index, model = build_index(repo_url)
    
    mode = input("Enter 'i' for interactive query or 'e' for evaluation (default is interactive): ").strip().lower()
    
    if mode == "e":
        run_evaluation(docs, embeddings, index, model)
    else:
        run_interactive(docs, embeddings, index, model)

if __name__ == "__main__":
    main()