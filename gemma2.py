import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def main() -> None:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python runtime. "
            "Make sure you're running on the GPU machine and in the right environment."
        )

    model_id = "google/gemma-2-9b-it"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()

    prompt = "Write me a poem about Machine Learning."
    inputs = tokenizer(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()


    # proc.userdata["llm"] = openai.LLM(
    #     model="google/gemma-2-9b-it",
    #     api_key=os.environ.get("gemma_api", "dummy"),
    #     base_url=os.environ.get("gemma_api", "http://192.168.30.239:8000/v1"),
    #     temperature=0.1,
    # )