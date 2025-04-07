import time
from openai import OpenAI
from transformers import pipeline, AutoTokenizer
from config import LLM_PROVIDER

# Pipelines
local_summarizer = None
custom_summarizer = None
client = OpenAI()

def init_local_summarizer():
    global local_summarizer
    if local_summarizer is None:
        print("Initializing local summarization model (facebook/bart-large-cnn)...")
        local_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return local_summarizer

def init_custom_summarizer():
    global custom_summarizer
    if custom_summarizer is None:
        print("Initializing custom summarization provider (t5-base)...")
        custom_summarizer = pipeline("summarization", model="t5-base")
    return custom_summarizer

def generate_summary(text, provider=LLM_PROVIDER):
    start_time = time.time()
    if provider == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes code snippets."},
                {"role": "user", "content": f"Summarize concisely the following code snippet:\n\n{text}"}
            ],
            temperature=0.5,
            max_completion_tokens=150,
        )
        summary = response.choices[0].message.content.strip()
    elif provider == "custom":
        summarizer = init_custom_summarizer()
        
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        model = summarizer.model
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        summarizer = init_local_summarizer()
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        model = summarizer.model
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    latency = time.time() - start_time
    return summary, latency
