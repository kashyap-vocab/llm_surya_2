import asyncio
import os
import time
import uuid

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional


# ---------- env loader ----------
def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_dotenv()
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")

MODEL_ID = "google/gemma-4-E4B-it"


# ---------- 4-bit config ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# ---------- load model ----------
print(f"Loading model {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=hf_token,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=hf_token,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

model.eval()

# 🔍 DEBUG: confirm quantization
print("is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", False))
print("is_loaded_in_8bit:", getattr(model, "is_loaded_in_8bit", False))

print("Model loaded.")


# ---------- app ----------
app = FastAPI(title="Gemma 4 OpenAI-compatible API")


# ---------- middleware ----------
@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[LATENCY] {request.method} {request.url.path} {elapsed_ms:.2f} ms")
    return response


# ---------- schemas ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = False


# ---------- generation ----------
def _generate(messages, max_new_tokens, temperature, top_p, enable_thinking):
    raw_messages = [{"role": m.role, "content": m.content} for m in messages]

    text = tokenizer.apply_chat_template(
        raw_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            use_cache=True,   # keep this for speed
        )

    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------- streaming ----------
def _stream_words(text: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    for word in text.split(" "):
        yield (
            f'data: {{"id":"{chunk_id}","object":"chat.completion.chunk",'
            f'"choices":[{{"delta":{{"content":"{word} "}},"index":0,"finish_reason":null}}]}}\n\n'
        )

    yield (
        f'data: {{"id":"{chunk_id}","object":"chat.completion.chunk",'
        f'"choices":[{{"delta":{{}},"index":0,"finish_reason":"stop"}}]}}\n\n'
    )
    yield "data: [DONE]\n\n"


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        _generate,
        req.messages,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.enable_thinking,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[GENERATION] took {elapsed_ms:.2f} ms")

    if req.stream:
        return StreamingResponse(_stream_words(text), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


# ---------- run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)