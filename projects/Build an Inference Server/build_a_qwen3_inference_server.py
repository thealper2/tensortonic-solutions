import json
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

MODEL_ID = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

print(f"Loaded {MODEL_ID}")

def prepare_inputs(prompt: str):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
    )

    assert messages, "Add the prompt messages"
    assert inputs is not None, "Render the prompt with the tokenizer"
    return {key: value.to(model.device) for key, value in inputs.items()}

def select_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_k is not None:
        top_values, top_indices = torch.topk(scaled_logits, top_k, dim=-1)
        cutoff = top_values[:, -1:]
        scaled_logits = torch.where(scaled_logits >= cutoff, scaled_logits, float('-inf'))

    probabilities = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    assert next_token is not None, "Sample one token"
    return next_token

@torch.inference_mode()
def generate_uncached(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> dict:
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    first_token_at = None

    sync_gpu()
    started_at = time.perf_counter()
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # TODO 3: select and append the next token, extend the mask, and stop on EOS.
        next_token = select_next_token(
            outputs.logits[:, -1, :],
            temperature,
            top_k,
        )
        if first_token_at is None:
            sync_gpu()
            first_token_at = time.perf_counter()

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=model.device),
        ], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    sync_gpu()
    finished_at = time.perf_counter()
    completion_ids = generated_ids[0, prompt_length:]
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids.tolist(),
        "input_tokens": prompt_length,
        "ttft_ms": (first_token_at - started_at) * 1000,
        "latency_ms": (finished_at - started_at) * 1000,
    }

@torch.inference_mode()
def generate_cached(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> dict:
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    first_token_at = None

    sync_gpu()
    started_at = time.perf_counter()
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = select_next_token(
            outputs.logits[:, -1, :],
            temperature,
            top_k,
        )
        assert next_token is not None, "Select the next token"
        if first_token_at is None:
            sync_gpu()
            first_token_at = time.perf_counter()

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=model.device),
        ], dim=1)
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    sync_gpu()
    finished_at = time.perf_counter()
    completion_ids = generated_ids[0, prompt_length:]
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids.tolist(),
        "input_tokens": prompt_length,
        "ttft_ms": (first_token_at - started_at) * 1000,
        "latency_ms": (finished_at - started_at) * 1000,
    }

def summarize_result(result: dict, used_kv_cache: bool) -> dict:
    output_tokens = len(result['token_ids'])
    output_tokens_per_second = output_tokens / (result['latency_ms'] / 1000) if result['latency_ms'] > 0 else 0.0

    return {
        "text": result["text"],
        "ttft_ms": round(result["ttft_ms"], 2),
        "latency_ms": round(result["latency_ms"], 2),
        "input_tokens": result["input_tokens"],
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens_per_second, 2),
        "used_kv_cache": used_kv_cache,
    }

BENCHMARK_PROMPT = "Explain why the sky looks blue in two sentences."

print("Warming up the model", flush=True)
generate_cached("Say hello in one sentence.", max_new_tokens=2)

print("Running without KV cache", flush=True)
uncached_result = generate_uncached(BENCHMARK_PROMPT, max_new_tokens=24)
print("Running with KV cache", flush=True)
cached_result = generate_cached(BENCHMARK_PROMPT, max_new_tokens=24)

comparison = {
    "outputs_match": uncached_result["token_ids"] == cached_result["token_ids"],
    "without_cache": summarize_result(uncached_result, used_kv_cache=False),
    "with_cache": summarize_result(cached_result, used_kv_cache=True),
}
print("__TT_COMPARISON__=" + json.dumps(comparison), flush=True)

@torch.inference_mode()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
):
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]

    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    previous_text = ""

    for _ in range(max_new_tokens):
        next_token = select_next_token(
            outputs.logits[:, -1, :],
            temperature,
            top_k,
        )
        assert next_token is not None, "Select the next token"
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        current_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
        chunk = current_text[len(previous_text):]
        if chunk:
            yield chunk
        previous_text = current_text

        if next_token.item() == tokenizer.eos_token_id:
            break

        attention_mask = torch.cat([
            attention_mask,
            torch.ones(
                (1, 1),
                dtype=attention_mask.dtype,
                device=model.device,
            ),
        ], dim=1)
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt to generate from")
    max_new_tokens: int = Field(32, ge=1, description="Maximum number of tokens to generate")
    temperature: float = Field(0.0, ge=0.0, description="Sampling temperature (0 for greedy)")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling cutoff")

class GenerateResponse(BaseModel):
    text: str
    ttft_ms: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    output_tokens_per_second: float
    used_kv_cache: bool

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="tinyinference")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    result = generate_cached(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    )

@app.post("/stream")
def stream_text(request: GenerateRequest):
    return StreamingResponse(
        generate_stream(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        ),
        media_type='text/plain',
    )

from fastapi.testclient import TestClient

client = TestClient(app)

health_response = client.get('/health')
generation_response = client.post(
    '/generate',
    json={'prompt': 'Hello, World!', 'max_new_tokens': 10}
)

assert health_response.status_code == 200
assert health_response.json() == {"status": "ok"}
assert generation_response.status_code == 200
assert generation_response.json()["output_tokens"] > 0

print(json.dumps(generation_response.json(), indent=2))