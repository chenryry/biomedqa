import json, torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,BitsAndBytesConfig,DataCollatorForLanguageModeling,Trainer,DefaultDataCollator
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from sentence_transformers import SentenceTransformer
import faiss
import os
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from util import formatting_func, tokenize, retrieve_context, format_prompt, generate_answer_rag, extract_answer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open("ori_pqal.json", "r") as f:
    raw_data = json.load(f)

examples = []
contexts = []
for pid, entry in raw_data.items():
    short_answer = entry.get("final_decision", "").strip().lower()
    long_answer = entry.get("LONG_ANSWER", "").strip()

    if short_answer:
        if long_answer:
            answer = f"{short_answer}. {long_answer}"
        else:
            answer = short_answer

        examples.append({
            "prompt": f"""You are a biomedical research assistant.
Question: {entry['QUESTION']}
Context: {' '.join(entry['CONTEXTS'])}
Answer (state your answer and explain why):""",
            "completion": f" {answer}"
        })

        contexts.append(' '.join(entry['CONTEXTS']))


dataset = Dataset.from_list(examples)
dataset = dataset.train_test_split(test_size=0.1)

model_id = "model_id"

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

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenized_dataset = dataset.map(tokenize, batched=False)

training_args = TrainingArguments(
    output_dir="./gemma-pubmedqa-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()
model.save_pretrained("./gemma-pubmedqa-lora")
tokenizer.save_pretrained("./gemma-pubmedqa-lora")

