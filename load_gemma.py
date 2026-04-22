import os
import asyncio
import time

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional


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

MODEL_ID = "google/gemma-2-9b-it"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # use float16 compute by default on consumer GPUs; bfloat16 can be slower or unsupported
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading model {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=hf_token,
    quantization_config=quantization_config,
    # force full model onto a single GPU (set to 0). Change if you want auto-sharding.
    device_map={"": 0},
)
model.eval()
print("Model loaded.")

# --- basic quantization / placement diagnostics ---
try:
    import bitsandbytes as _bnb  # type: ignore
    has_bnb = True
except Exception:
    _bnb = None
    has_bnb = False

try:
    # count likely 4bit modules from bitsandbytes (class name can vary by version)
    bnb4_count = 0
    for m in model.modules():
        nm = m.__class__.__name__
        if "4bit" in nm.lower() or nm.lower().startswith("linear4bit") or "bnb" in nm.lower():
            bnb4_count += 1

    sample_param = next(model.parameters())
    print(
        f"bitsandbytes installed: {has_bnb}, linear-like 4bit module count: {bnb4_count},"
        f" sample param dtype: {sample_param.dtype}, device: {sample_param.device}"
    )
except Exception as e:
    print("Quantization diagnostics failed:", e)

try:
    from prompt_builder import build_prompt, word_stream, strip_response
    print("[cython] prompt_builder extension loaded")
except ImportError:
    print("[cython] extension not found, using pure-Python fallback")
    strip_response = str.strip

app = FastAPI(title="Gemma API")


# ---------- middleware for latency ----------

@app.middleware("http")
async def log_latency(request: Request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    print(
        f"[LATENCY] {request.method} {request.url.path} "
        f"completed in {latency_ms:.2f} ms"
    )

    return response


# ---------- schemas ----------

class Message(BaseModel):
    role: str   # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False


# ---------- helpers ----------

# Pure-Python fallbacks used when Cython extension is not compiled
try:
    build_prompt  # already imported from prompt_builder above
except NameError:
    def build_prompt(messages: List[Message]) -> str:
        parts = []
        for msg in messages:
            if msg.role in ("user", "system"):
                parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>\n")
            elif msg.role == "assistant":
                parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

try:
    word_stream  # already imported from prompt_builder above
except NameError:
    def word_stream(text: str):
        for word in text.split(" "):
            yield word + " "


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # use inference_mode for slightly better performance
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    generated = outputs[0][input_len:]
    return strip_response(tokenizer.decode(generated, skip_special_tokens=True))


def word_stream(text: str):
    for word in text.split(" "):
        yield word + " "


# ---------- routes ----------

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Raw text-in / text-out endpoint."""
    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, _generate, req.prompt, req.max_new_tokens, req.temperature, req.top_p
    )

    end = time.perf_counter()
    print(f"[GENERATION] /generate took {(end - start)*1000:.2f} ms")

    if req.stream:
        return StreamingResponse(
            (chunk for chunk in word_stream(text)),
            media_type="text/plain",
        )

    return {"response": text}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Multi-turn chat endpoint using Gemma instruct format."""
    start = time.perf_counter()

    prompt = build_prompt(req.messages)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, _generate, prompt, req.max_new_tokens, req.temperature, req.top_p
    )

    end = time.perf_counter()
    print(f"[GENERATION] /chat took {(end - start)*1000:.2f} ms")

    if req.stream:
        return StreamingResponse(
            (chunk for chunk in word_stream(text)),
            media_type="text/plain",
        )

    return {"role": "assistant", "content": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)