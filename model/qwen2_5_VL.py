import base64
import io,os
import torch
from outlines.models import transformers
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from openai import OpenAI
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from typing import Union, Tuple, List
# from transformers.models.qwen/
from qwen_agent.tools.base import BaseTool, register_tool
from transformers import AutoProcessor


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()

    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _system_button(self, button: str):
        raise NotImplementedError()

    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()


from IPython.display import display


model_id="qwen2.5-vl-7b-instruct"
model_path = "./qwen2.5VL_model"
# Prapare your screenshot file and global query
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
def run_for_case(instruction, img_path):
    screenshot = img_path

    # You can also start from an intermediate screenshot without passing in the history.
    user_query = f'The user query:点击最符合指令“{instruction}”的元素 (You have done the following operation on the current device):'

    # The resolution of the device will be written into the system prompt.
    dummy_image = Image.open(screenshot)
    resized_height, resized_width  = smart_resize(dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )
    def encode_image(image_path):
        img = Image.open(screenshot)
        # img.save("tmp.png")
        image_data = io.BytesIO()
        img.convert("RGB")
        img.save(image_data, format='JPEG')
        image_data_bytes = image_data.getvalue()
        encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
        return encoded_image

    base64_image = encode_image(screenshot)
    # Build messages

    system_message = NousFnCallPrompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
        ],
        functions=[mobile_use.function],
        lang=None,
    )

    system_message = system_message[0].model_dump()
    messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": msg["text"]} for msg in system_message["content"]
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": processor.image_processor.min_pixels,
                    "max_pixels": processor.image_processor.max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": user_query},
            ],
        }
    ]
    ## api method:
    # client = OpenAI(
    #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    #     api_key="sk-6b484602642e4543bc495c224c545e4f",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # # print(json.dumps(messages, indent=4))
    # try:
    #     completion = client.chat.completions.create(
    #         model = model_id,
    #         messages = messages,
    #
    #     )
    #     output_text = completion.choices[0].message.content
    #     action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    # except:
    #     action = {'name': 'ERROR', 'arguments': {'action': 'click', 'coordinate': [-1, -1]}}
    ## local method:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

    print(action)
    action['origin_size'] = (resized_width, resized_height)
    return action