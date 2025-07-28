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