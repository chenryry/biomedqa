import torch
import numpy as np
import faiss
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

model_id = "model_id"

def load_model_and_index():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    import json
    with open("ori_pqal.json", "r") as f:
        raw_data = json.load(f)

    contexts = [' '.join(entry['CONTEXTS']) for entry in raw_data.values()]
    corpus_embeddings = embedder.encode(contexts, convert_to_tensor=False, show_progress_bar=True)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    return model, tokenizer, index, embedder, contexts

def embed_new_doc(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        new_text = f.read()

    new_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    new_embedding = new_embedder.encode([new_text])
    new_embedding = new_embedding / np.linalg.norm(new_embedding, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(new_embedding.shape[1])
    index.add(new_embedding)

def get_context_and_answer(query, model, tokenizer, index, embedder, contexts):
    return generate_answer_rag(query, model, tokenizer, index, embedder, contexts)
def formatting_func(example):
    return example["prompt"] + example["completion"]


def tokenize(example):
    text = formatting_func(example)
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

def retrieve_context(question, top_k=1):
    q_embed = embedder.encode([question])
    q_embed = q_embed / np.linalg.norm(q_embed, axis=1, keepdims=True)
    D, I = index.search(np.array(q_embed), top_k)
    return contexts[I[0][0]]

def format_prompt(question, context):
    return f"""You are a biomedical research assistant.\nQuestion: {question}\nContext: {context}\nAnswer (state your answer and explain why):"""

def generate_answer_rag(question):
    context = retrieve_context(question)
    prompt = format_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.split("Answer")[-1].strip(":").strip()

def extract_answer(text):
    text = text.lower()
    if "yes" in text and "no" not in text:
        return "yes"
    elif "no" in text and "yes" not in text:
        return "no"
    elif "maybe" in text:
        return "maybe"
    return "unknown"