import argparse
import os
import re
from io import BytesIO
from typing import List, Literal, Optional

import requests
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

import json

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# System Message
grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."

# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

assistant_template = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}{{'\n' + message['content'][0]['text']}}{% if 'end_turn' in message and message['end_turn'] %}{{'<|diff_marker|>\n'}}{% else %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|recipient|>' }}{% endif %}"

# Special Tokens
additional_special_tokens = [
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|recipient|>",
    "<|diff_marker|>",
]

# Plugin Functions
select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

agent_system_message = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""

user_instruction = """Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal}

Previous actions:
{previous_actions}
"""

until = ["<|diff_marker|>"]


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


model = Qwen2VLForConditionalGeneration.from_pretrained(os.environ['model_path'],
                                                       device_map="auto")
processor = Qwen2VLProcessor.from_pretrained(os.environ['model_path'],max_pixels = 46 * 26 * 28 * 28)
tokenizer = processor.tokenizer
def load_pretrained_model(model_path):
    # model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
    # processor = Qwen2VLProcessor.from_pretrained(model_path)
    # tokenizer = processor.tokenizer
    return model, processor, tokenizer


def generate_response(
        model,
        processor,
        tokenizer,
        image: Image.Image,
        instruction: str,
        previous_actions: Optional[str | List[str]] = None,
        low_level_instruction: Optional[str] = None,
        mode: Literal["self-plan", "force-plan", "grounding"] = "self-plan",
        temperature: float = 0,
        max_new_tokens: int = 1024,
):
    system_message = {
        "role": "system",
        "content": grounding_system_message if mode == "grounding" else agent_system_message,
    }

    if isinstance(previous_actions, list):
        # Convert previous actions to string. Expecting the format:
        # ["Step 1: Swipe up", "Step 2: Click on the search bar"]
        previous_actions = "\n".join(previous_actions)
    if not previous_actions:
        previous_actions = "None"
    user_message = {
        "role": "user",
        "content": user_instruction.format(
            overall_goal=instruction,
            previous_actions=previous_actions,
            low_level_instruction=low_level_instruction,
        ),
    }

    if low_level_instruction:
        # If low-level instruction is provided
        # We enforce using "Action: {low_level_instruction} to guide generation"
        recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
    elif mode == "force-plan":
        recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
    elif mode == "grounding":
        recipient_text = "<|im_start|>assistant<|recipient|>all\n"
    elif mode == "self-plan":
        recipient_text = ""
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Generate response
    messages = [system_message, user_message]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template
    )
    text += recipient_text
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    cont = model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens)

    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]):]
    text_outputs = tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    for term in until:
        if len(term) > 0:
            text_outputs = text_outputs.split(term)[0]
    return text_outputs


def main(args):
    model, processor, tokenizer = load_pretrained_model(args.model_path)
    # model.to(args.device)
    model.tie_weights()
    image = load_image(args.image_path)
    response = generate_response(
        model,
        processor,
        tokenizer,
        image,
        args.instruction,
        args.previous_actions,
        args.low_level_instruction,
        args.mode,
        args.temperature,
    )
    return (response)


def extract_coordinates(click_command):
    # 正则表达式模式用于匹配x和y的值
    pattern = r'pyautogui\.click$x=(.*?), y=(.*?)$'

    match = re.search(pattern, click_command)
    if match:
        # 如果找到匹配项，则提取x和y的值，并尝试将它们转换为整数
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    else:
        raise ValueError("输入的字符串格式不正确")
def run_for_case(instruction, img_path):
    class DotDict(dict):
        def __init__(self, *args, **kwargs):
            super(DotDict, self).__init__(*args, **kwargs)

        def __getattr__(self, key):
            value = self[key]
            if isinstance(value, dict):
                value = DotDict(value)
            return value

    args = {
        "model_path": os.environ["model_path"],
        "device": "cuda",
        "image_path": img_path,
        "instruction": instruction,
        "previous_actions": None,
        "low_level_instruction": None,
        "mode": "self-plan",
        "temperature": 0,
        "max_new_tokens": 1024,
    }
    return main(DotDict(args))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--image_path", type=str, default="examples/AndroidControl.png")
    # parser.add_argument(
    #     "--instruction",
    #     type=str,
    #     default="In the BBC News app , Turn ON the news alert notification for the BBC News app.",
    # )
    # parser.add_argument("--previous_actions", type=str, required=False)
    # parser.add_argument("--low_level_instruction", type=str, required=False)
    # parser.add_argument("--mode", type=str, default="self-plan")
    # parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--max_new_tokens", type=int, default=1024)
    # args = parser.parse_args()
    # main(args)
    print(run_for_case(instruction="前往逛逛", img_path="screenshots/android/1-1-1-v10.0.0.jpg"))
