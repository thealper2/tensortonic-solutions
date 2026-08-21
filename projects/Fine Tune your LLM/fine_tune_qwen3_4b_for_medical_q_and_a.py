from collections import Counter
import json
import os
import re

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

if not torch.cuda.is_available():
    raise RuntimeError("A GPU is required for this project")

device = torch.device("cuda")
print(torch.cuda.get_device_name(0))
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

WORKSPACE = "/tmp/tensortonic"
DATA_DIR = os.path.join(WORKSPACE, "data", "medquad")
MODEL_DIR = os.path.join(WORKSPACE, "models")
MODEL_CACHE = os.path.join(MODEL_DIR, "huggingface")
DATASET_CACHE = os.path.join(WORKSPACE, "cache", "datasets")

os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(DATASET_CACHE, exist_ok=True)

print(WORKSPACE)

MODEL_ID = "Qwen/Qwen3-4B"
print(f"Downloading {MODEL_ID} to {MODEL_CACHE}", flush=True)
local_model_path = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE)
print(f"Model files ready at {local_model_path}", flush=True)

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print(quantization)

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=quantization,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)
print(f"Loaded {MODEL_ID}")

DATASET_ID = "Hmehdi515/MedQuad"
DATASET_FILES = {
    "train": "https://huggingface.co/datasets/Hmehdi515/MedQuad/resolve/main/medquad_train.csv",
    "validation": "https://huggingface.co/datasets/Hmehdi515/MedQuad/resolve/main/medquad_val.csv",
}

print(f"Downloading {DATASET_ID}", flush=True)
source = load_dataset("csv", data_files=DATASET_FILES, cache_dir=DATASET_CACHE)

os.makedirs(DATA_DIR, exist_ok=True)
raw_train_path = os.path.join(DATA_DIR, "raw_train.jsonl")
raw_validation_path = os.path.join(DATA_DIR, "raw_validation.jsonl")
source["train"].to_json(raw_train_path)
source["validation"].to_json(raw_validation_path)

print(f"Saved {len(source['train'])} rows to {raw_train_path}", flush=True)
print(f"Saved {len(source['validation'])} rows to {raw_validation_path}", flush=True)
print(source)

columns = source['train'].column_names
answer_lengths = [len(str(row['answer']).split()) for row in source['train']]
median_answer_words = sorted(answer_lengths)[len(answer_lengths) // 2]

assert {"question", "answer"}.issubset(columns), "Check the dataset columns"
assert answer_lengths, "Calculate the answer lengths"
assert median_answer_words is not None, "Calculate the median answer length"

print({
    "columns": columns,
    "training_rows": len(source["train"]),
    "median_answer_words": median_answer_words,
})
print(source["train"][0])

def select_examples(split, limit):
    rows = []
    for example in source[split]:
        question = str(example["question"]).strip()
        answer = str(example["answer"]).strip()
        answer_words = len(answer.split())

        keep_example = question and 20 <= answer_words <= 160
        if keep_example:
            rows.append({"question": question, "answer": answer})
        if len(rows) == limit:
            break
    return Dataset.from_list(rows)

print("Creating a small training set and a separate evaluation set", flush=True)
dataset = DatasetDict({
    "train": select_examples("train", 96).shuffle(seed=42),
    "evaluation": select_examples("validation", 8),
})

print(f"Writing dataset files to {DATA_DIR}", flush=True)
dataset["train"].to_json(os.path.join(DATA_DIR, "train.jsonl"))
dataset["evaluation"].to_json(os.path.join(DATA_DIR, "evaluation.jsonl"))
dataset.save_to_disk(os.path.join(DATA_DIR, "arrow"))

SYSTEM_PROMPT = (
    "Answer the medical question clearly and concisely. When a question "
    "requires personal medical advice, recommend consulting a qualified clinician."
)
print(dataset)
print(dataset["train"][0])

def generate_answer(model, question):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
    )

    assert messages, "Add the prompt messages"
    assert inputs is not None, "Render the prompt with the tokenizer"
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

def answer_f1(prediction, reference):
    predicted_tokens = prediction.lower().split()
    reference_tokens = reference.lower().split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    pred_counter = Counter(predicted_tokens)
    ref_counter = Counter(reference_tokens)
    overlap = sum((pred_counter & ref_counter).values())

    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate_model(model, rows):
    model.eval()
    results = []
    for row in rows:
        prediction = generate_answer(model, row["question"])
        score = answer_f1(prediction, row["answer"])
        results.append({
            "question": row["question"],
            "reference": row["answer"],
            "prediction": prediction,
            "token_f1": round(score * 100, 2),
        })
    return {
        "token_f1": round(sum(row["token_f1"] for row in results) / len(results), 2),
        "examples": len(results),
        "results": results,
    }

baseline = evaluate_model(base_model, dataset["evaluation"])
print(json.dumps({
    "token_f1": baseline["token_f1"],
    "examples": baseline["examples"],
}, indent=2))

def format_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

train = dataset["train"].map(format_example)
assert train[0]["text"].strip(), "The formatted training text is empty"
print(train[0]["text"])

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    task_type="CAUSAL_LM",
)
print(f"LoRA rank: {lora.r}")

training_config = SFTConfig(
    output_dir="/tmp/qwen-medquad",
    learning_rate=0.0002,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_length=384,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    bf16=True,
    gradient_checkpointing=True,
    dataset_text_field="text",
    seed=42,
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=training_config,
    train_dataset=train,
    peft_config=lora,
)
train_result = trainer.train()

assert trainer is not None, "Create the trainer"
assert train_result is not None, "Start the training run"
print({"training_loss": round(train_result.training_loss, 5)})

trainer.model.eval()
fine_tuned_result = evaluate_model(trainer.model, dataset["evaluation"])

score_change = fine_tuned_result["token_f1"] - baseline["token_f1"]
weakest = sorted(fine_tuned_result["results"], key=lambda x: x["token_f1"])[:2]

print(json.dumps({
    "baseline_token_f1": baseline["token_f1"],
    "fine_tuned_token_f1": fine_tuned_result["token_f1"],
    "score_change": score_change,
    "weakest_answers": weakest,
}, indent=2))