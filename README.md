# vLLM + Wrapper Deployment — Gemma 2-9b-it on GCP

A production inference stack serving **Google Gemma 2-9b-it** via vLLM, fronted by a FastAPI proxy wrapper that handles identity management, probe detection, response sanitization, and structured logging.

---

## Architecture

```
Client (OpenAI SDK / curl)
        │
        │  POST /v1/chat/completions  (port 9000, public)
        ▼
┌─────────────────────────┐
│     Wrapper (surya-api) │   FastAPI proxy — hides real model, detects probes,
│     dronavocab/llm_     │   sanitizes responses, logs JSON, supports streaming
│     surya2:wrapper      │
└────────────┬────────────┘
             │  proxies to http://llm-backend:7000/v1/chat/completions
             ▼
┌─────────────────────────┐
│   vLLM Backend          │   OpenAI-compatible inference server
│   dronavocab/llm_       │   Model: google/gemma-2-9b-it (4-bit quantized)
│   surya2:vllm           │   Port: 7000 (internal only)
└─────────────────────────┘
```

**Public model name exposed to clients:** `surya-02`
**Real model running underneath:** `google/gemma-2-9b-it`

---

## Prerequisites

- GCP VM with NVIDIA GPU (e.g., `n1-standard-8` + T4 or A100)
- CUDA drivers installed on the VM
- Docker + Docker Compose installed
- HuggingFace account with access to [Gemma](https://huggingface.co/google/gemma-2-9b-it) (gated model — requires acceptance of license)
- Docker Hub account (for pushing/pulling images)

---

## Project Files

| File | Purpose |
|---|---|
| `Dockerfile` | vLLM backend image based on `vllm/vllm-openai:latest` |
| `Dockerfile.wrapper` | 2-stage Cython-compiled wrapper image |
| `docker-compose.yml` | Orchestrates both services with healthchecks |
| `wrapper.py` | FastAPI proxy (compiled to `.so` — not included in final image) |
| `wrapper_entrypoint.py` | Entrypoint that imports the compiled wrapper module |
| `setup_wrapper.py` | Cython build configuration |
| `requirements-docker.txt` | Extra deps for vLLM image (`bitsandbytes`, `accelerate`) |
| `requirements-wrapper.txt` | Wrapper deps (`cython`, `fastapi`, `uvicorn`, `httpx`) |
| `.env` | Local secrets — `MODEL_ID` and `HF_TOKEN` (git-ignored) |

---

## Environment Setup

Create a `.env` file in the project root:

```env
MODEL_ID=google/gemma-2-9b-it
HF_TOKEN=hf_your_token_here
```

> **Never commit `.env` to git.** It is already listed in `.gitignore`.

---

## vLLM Command

This is the exact command used to start vLLM on the GCP VM:

```bash
vllm serve google/gemma-2-9b-it \
    --port 9000 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.6 \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic
```

### Flag Reference

| Flag | Value | Explanation |
|---|---|---|
| `--port` | `9000` | Port vLLM listens on |
| `--quantization` | `bitsandbytes` | Activates 4-bit NF4 quantization via bitsandbytes — reduces VRAM requirement significantly |
| `--load-format` | `bitsandbytes` | Loads model weights in quantized format instead of full precision |
| `--gpu-memory-utilization` | `0.6` | Reserves 60% of total GPU VRAM for the model's KV cache and weights |
| `--enable-auto-tool-choice` | — | Enables OpenAI-compatible function/tool calling |
| `--tool-call-parser` | `pythonic` | Uses pythonic tool call format — required for Gemma models |

> If you have more VRAM available, you can increase `--gpu-memory-utilization` to `0.7` or `0.8` for better throughput.

---

## Wrapper (wrapper.py)

The wrapper is a FastAPI reverse proxy that intercepts all client requests before they reach vLLM.

### What it does

**1. Identity hiding**
- Clients see model name `surya-02` in `/v1/models` and responses
- Internally, requests are forwarded using `google/gemma-2-9b-it`
- A system guard prompt is prepended to every conversation:
  ```
  You are a helpful assistant. Your name is surya-02.
  If someone asks about your architecture, training, or creator,
  respond that you are Surya AI and cannot share further details.
  ```

**2. Probe detection**
- Detects attempts to identify the underlying model using patterns like:
  - `"what model are you"`, `"which model"`, `"model name"`, `"who are you"`, `"system prompt"`, etc.
- Returns a canned response without forwarding the request to vLLM

**3. Response sanitization**
- Scans every response from vLLM and strips any references to GPT, OpenAI, gpt-oss, etc. using regex

**4. Streaming support**
- Transparently proxies `stream: true` requests — SSE chunks are forwarded directly to the client

**5. Structured logging**
- Every request/response logged as JSON to stdout and `/app/logs/requests.log`
- Logs include: request ID, client IP, probe flag, model, parameters, response content, latency (ms)
- Log rotation: 10 MB per file, 5 backups retained

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint |
| `GET` | `/v1/models` | Returns only `surya-02` |
| `GET` | `/health` | Liveness check |

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `INTERNAL_LLM_URL` | `http://llm-backend:7000/v1/chat/completions` | vLLM upstream URL |
| `INTERNAL_MODEL` | `google/gemma-2-9b-it` | Real model ID forwarded to vLLM |
| `PUBLIC_MODEL_NAME` | `surya-02` | Model name shown to clients |
| `BACKEND_IDENTITY` | `Surya AI` | Identity claimed when probed |
| `WRAPPER_PORT` | `9000` | Port the wrapper listens on |
| `LOG_FILE` | `/app/logs/requests.log` | Log file path inside container |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Cython compilation

`wrapper.py` is compiled to a native `.so` shared library using Cython inside `Dockerfile.wrapper` (Stage 1). The source files (`wrapper.py`, `wrapper.c`, `setup_wrapper.py`) are deleted before the final image is assembled — only the binary `.so` and `wrapper_entrypoint.py` are included. This prevents source inspection of the proxy logic.

---

## Docker: Build & Push

### vLLM Backend

```bash
# Build
docker build -t dronavocab/llm_surya2:vllm -f Dockerfile .

# Push to Docker Hub
docker push dronavocab/llm_surya2:vllm
```

### Wrapper

```bash
# Build (2-stage: compiles wrapper.py with Cython, then strips source)
docker build -t dronavocab/llm_surya2:wrapper -f Dockerfile.wrapper .

# Push
docker push dronavocab/llm_surya2:wrapper
```

> Personal registry alternative (used during development):
> ```bash
> docker build -t kashyapsai2003/gemma_inhouse:vllm -f Dockerfile .
> docker build -t kashyapsai2003/gemma_inhouse:wrapper -f Dockerfile.wrapper .
> ```

---

## GCP Deployment

### 1. Create the VM

In GCP Console or via `gcloud`:

```bash
gcloud compute instances create surya-llm-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

### 2. Install CUDA drivers

```bash
# On the VM
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
```

### 3. Install Docker + NVIDIA Container Toolkit

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Clone the repo and configure

```bash
git clone <your-repo-url>
cd VLLMS

# Create .env
echo "MODEL_ID=google/gemma-2-9b-it" > .env
echo "HF_TOKEN=hf_your_token_here" >> .env
```

### 5. Pull images and start

```bash
docker-compose pull
docker-compose up -d
```

### 6. Open firewall port (GCP)

```bash
gcloud compute firewall-rules create allow-surya-api \
    --allow tcp:9000 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow vLLM wrapper API"
```

---

## Running with Docker Compose

```bash
# Start all services (detached)
docker-compose up -d

# Check service status
docker-compose ps

# Stream logs from both services
docker-compose logs -f

# Stream logs from wrapper only
docker-compose logs -f surya-api

# Stream logs from vLLM only
docker-compose logs -f llm-backend

# Stop and remove containers
docker-compose down
```

---

## Health Checks

| Service | Endpoint | Startup Window | Interval |
|---|---|---|---|
| vLLM backend | `GET http://localhost:7000/health` | 10 minutes | 30s |
| Wrapper | `GET http://localhost:9000/health` | 10 seconds | 15s |

The wrapper uses `depends_on: condition: service_healthy` — it will not start until the vLLM backend passes its healthcheck.

---

## API Usage

The wrapper is OpenAI-compatible. Point any OpenAI SDK client at `http://<VM_EXTERNAL_IP>:9000`.

### curl example

```bash
curl http://<VM_EXTERNAL_IP>:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "surya-02",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ],
        "temperature": 0.7
    }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<VM_EXTERNAL_IP>:9000/v1",
    api_key="not-required"
)

response = client.chat.completions.create(
    model="surya-02",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### List available models

```bash
curl http://<VM_EXTERNAL_IP>:9000/v1/models
```

---

## Volumes

| Volume | Mount path | Purpose |
|---|---|---|
| `model_cache` | `/root/.cache/huggingface` (vllm-backend) | Persists downloaded model weights across restarts |
| `wrapper_logs` | `/app/logs` (surya-api) | Persists request/response logs |

---

## Quantization Details

The model runs in **4-bit NF4 quantization** via bitsandbytes:

- Quantization type: NormalFloat 4-bit (NF4)
- Double quantization enabled (quantizes the quantization scalars)
- Compute dtype: float16

This reduces the Gemma 2-9b-it memory footprint from ~18 GB (full precision) to ~6–7 GB, making it fit on a single T4 GPU.
