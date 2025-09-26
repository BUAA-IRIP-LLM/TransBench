import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default: Load the model on the available device(s)

model_path = os.environ['model_path']
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path, max_pixels = 1024*1024)


def run_for_case(instruction, img_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": "In this UI screenshot, what is the position of the element "
                                         "corresponding to the command \"{}\" (with bbox)?".format(instruction)},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text[0]
    # <|object_ref_start|>language switch<|object_ref_end|><|box_start|>(576,12),(592,42)<|box_end|><|im_end|>

if __name__ == '__main__':
    print(run_for_case(instruction="前往逛逛", img_path="screenshots/android/1-1-1-v10.0.0.jpg"))
