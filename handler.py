import base64
from io import BytesIO

import runpod
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "allenai/Molmo2-8B"

model = None
processor = None


def load_model():
    global model, processor

    if model is not None and processor is not None:
        return

    print(f"Loading model {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    print("Model loaded.")


def decode_image(image_b64: str):
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img


def handler(event):
    """
    Expected input JSON:
    {
      "input": {
        "prompt": "Describe this image",
        "image": "<base64-encoded image>"  # optional
      }
    }
    """
    load_model()

    inputs = event.get("input", {}) or {}
    prompt = inputs.get("prompt", "")
    image_b64 = inputs.get("image", None)

    if not isinstance(prompt, str):
        return {"error": "prompt must be a string"}

    image = None
    if image_b64:
        try:
            image = decode_image(image_b64)
        except Exception as e:
            return {"error": f"Failed to decode image: {e}"}

    # Prepare multimodal inputs
    if image is not None:
        proc_inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)
    else:
        proc_inputs = processor(
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **proc_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

    # Molmo processor may have custom decoding
    try:
        response = processor.batch_decode(
            generated,
            skip_special_tokens=True
        )[0]
    except Exception:
        response = processor.decode(
            generated[0],
            skip_special_tokens=True
        )

    return {"output": response}


runpod.serverless.start({"handler": handler})
