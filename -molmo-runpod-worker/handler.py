import base64
from io import BytesIO

import runpod
import torch
from PIL import Image
from transformers import AutoProcessor, MolmoForCausalLM

MODEL_ID = "allenai/Molmo2-8B"

model = None
processor = None


def load_model():
    global model, processor
    if model is not None and processor is not None:
        return
    model = MolmoForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )


def handler(job):
    global model, processor

    data = job.get("input", {}) or {}
    if data.get("_test") or data.get("test"):
        return {"response": "ok", "test": True}

    if model is None or processor is None:
        load_model()

    prompt = data.get("prompt", "")
    image_b64 = data.get("image")

    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor(
            text=prompt,
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )

    text = processor.decode(output_ids[0], skip_special_tokens=True)
    return {"response": text}


runpod.serverless.start({"handler": handler})
