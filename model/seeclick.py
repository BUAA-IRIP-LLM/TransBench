import os

# from resource import *

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

model_path = os.environ['model_path'] if "model_path" in os.environ else "seeclick_model"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


def run_for_case(instruction, img_path):
    prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
    # prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"  # Use this prompt for generating bounding box
    query = tokenizer.from_list_format([
        {'image': img_path},  # Either a local path or an url
        {'text': prompt.format(instruction)},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    response = eval(response)
    return response

if __name__ == '__main__':
    print(run_for_case(instruction="前往逛逛", img_path="screenshots/android/1-1-1-v10.0.0.jpg"))
