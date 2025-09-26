import argparse
import torch
import os
import json
from tqdm import tqdm
import time
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor
import ast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = os.environ['model_path']
tuned_model_path = os.environ['tuned_model_path']

model = AutoModelForCausalLM.from_pretrained(
    tuned_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def run_for_case(instruction, img_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", 'text': None},
                {
                    "type": "text",
                    "text": "Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: " + instruction,
                }
            ],
        }
    ]
    image_file = img_path
    image = Image.open(image_file).convert("RGB")

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            # do_sample=True,
            # temperature=0.9,
        )


    output_ids = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(output_ids, skip_special_tokens=True)
    print(response)

    coords = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())

    return [x / 1000.0 for x in coords]
