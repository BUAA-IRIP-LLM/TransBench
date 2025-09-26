# need:
# python -m vllm.entrypoints.openai.api_server --served-model-name osunlp/UGround-V1-7B --model osunlp/UGround-V1-7B --dtype float16
import base64
import io

from PIL import Image
def format_openai_template(description: str, base64_image):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {description}

Answer:"""
                },
            ],
        },
    ]
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def encode_local_img_to_base64_jpeg(imgpath):
    img = Image.open(imgpath)
    img = img.convert('RGB')
    image_data = io.BytesIO()
    img.save(image_data, format='JPEG')
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
    return encoded_image

def run_for_case(instruction, img_path):

    messages = format_openai_template(instruction,
                                      encode_local_img_to_base64_jpeg(img_path))

    completion = client.chat.completions.create(
        model='osunlp/UGround-V1-7B',
        messages=messages,
        temperature=0
        # REMEMBER to set temperature to ZERO!
        # REMEMBER to set temperature to ZERO!
        # REMEMBER to set temperature to ZERO!
    )
    tmp = eval(completion.choices[0].message.content)
    return [x/1000.0 for x in tmp]

# The output will be in the range of [0,1000), which is compatible with the original Qwen2-VL
# So the actual coordinates should be (x/1000*width, y/1000*height)
if __name__ == '__main__':
    print(run_for_case(instruction="前往逛逛", img_path="screenshots/android/1-1-1-v10.0.0.jpg"))