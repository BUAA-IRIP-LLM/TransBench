import os
# from resource import *

os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from PIL import Image, ImageDraw
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ast

model_path = os.environ['model_path']

llm = LLM(
    model=model_path,
    tokenizer_mode="slow",
    dtype="bfloat16",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False
)

def run_for_case(instruction, img_path):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: " + instruction,
                }
            ],
        }
    ]

    message = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    outputs = llm.generate(
        {
            "prompt_token_ids": message,
            "multi_modal_data": {
                "image": [
                    Image.open(img_path),
                ],
                "max_image_size": 980,  # [Optional] The max image patch size, default `980`
                "split_image": True,  # [Optional] whether to split the images, default `True`
            },
        },
        sampling_params=SamplingParams(max_tokens=50, top_k=1, stop=["<|im_end|>"]),
    )

    for o in outputs:
        generated_tokens = o.outputs[0].token_ids
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(response)
        coords = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
        return [x/1000.0 for x in coords]

if __name__ == "__main__":
    run_for_case(instruction="前往逛逛", img_path="../../screenshots/android/1-1-1-v10.0.0.jpg")

