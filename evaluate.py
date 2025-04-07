import json
import time
import os
from retrieve import retrieve_and_summarize

def load_test_data(test_file_path):
    with open(test_file_path, 'r') as f:
        return json.load(f)

def normalize_path(file_path):
    # Convert to relative path if it is an absolute path
    normalized_path = os.path.relpath(file_path).replace(os.sep, '/').lower()
    normalized_path = normalized_path.replace("repo/", "")
    return normalized_path

def evaluate(test_file_path, docs, index, model, top_k=10):
    test_data = load_test_data(test_file_path)

    total_files = 0
    correct_files = 0
    latency_details = []

    for test_case in test_data:
        query = test_case['question']
        expected_files = [file for file in test_case['files']]

        start_time = time.time()

        retrieved_results = retrieve_and_summarize(query, docs, index, model, top_k=top_k, generate_summaries=False)

        latency = time.time() - start_time
        latency_details.append(latency)

        retrieved_filenames = [normalize_path(doc['filename']) for doc in retrieved_results]
        
        # Check how many of the expected files are in the retrieved results
        correct_files_in_case = len([f for f in expected_files if f in retrieved_filenames])
        total_files_in_case = len(expected_files)
        
        total_files += total_files_in_case
        correct_files += correct_files_in_case

        print(f"Query: {query}")
        print(f"Expected files: {expected_files}")
        print(f"Retrieved filenames: {retrieved_filenames}")
        print(f"Correct files retrieved: {correct_files_in_case}/{total_files_in_case}")
        print(f"Latency: {latency:.4f} seconds")

    # Calculate Precision and Recall
    precision = correct_files / total_files if total_files > 0 else 0
    recall = correct_files / total_files if total_files > 0 else 0

    avg_latency = sum(latency_details) / len(latency_details) if latency_details else 0

    print(f"\nEvaluation Summary:")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Average Latency: {avg_latency:.4f} seconds")

    return precision, recall, avg_latency
