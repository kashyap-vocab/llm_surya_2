# ─────────────────────────────────────────────────────────────────────────────
# vLLM Backend Image
# Model name is NOT baked in — injected at runtime via .env / docker-compose
# Push: docker push kashyapsai2003/gemma_inhouse:vllm
# ─────────────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

WORKDIR /app

EXPOSE 7000

# Override the base image ENTRYPOINT (vllm/vllm-openai sets it to "vllm serve")
# so our shell-form CMD runs cleanly and $MODEL_ID is expanded from .env
ENTRYPOINT []
CMD vllm serve "$MODEL_ID" \
        --port 7000 \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --gpu-memory-utilization 0.7 \
        --hf-token "$HF_TOKEN"
