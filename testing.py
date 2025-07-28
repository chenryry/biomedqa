import json
from util import extract_answer
from sklearn.metrics import accuracy_score, classification_report

model = 'model'
tokenizer = 'tokenizer'

with open("test_ground_truth.json") as f:
    ground_truth = json.load(f)

with open("ori_pqal.json") as f:
    test_dataset = json.load(f)

from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

gemma_outputs = {}

for example_id, example in test_dataset.items():
    prompt = example["QUESTION"] + "\n" + "\n".join(example["CONTEXTS"])
    output = pipe(prompt, max_new_tokens=10)[0]["generated_text"]
    gemma_outputs[example_id] = output



common_keys = list(set(gemma_outputs.keys()) & set(ground_truth.keys()))

gemma_preds = {k: extract_answer(gemma_outputs[k]) for k in common_keys}
gt_labels = {k: ground_truth[k].lower() for k in common_keys}


filtered_keys = [k for k in common_keys if gemma_preds[k] in {"yes", "no", "maybe"}]

y_true = [gt_labels[k] for k in filtered_keys]
y_pred = [gemma_preds[k] for k in filtered_keys]

gemma_acc = accuracy_score(y_true, y_pred)
print("Gemma Accuracy:", gemma_acc)

print("Gemma Report:")
print(classification_report(y_true, y_pred, labels=["yes", "no", "maybe"]))