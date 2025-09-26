import shutil

import argparse


parser = argparse.ArgumentParser(description='Example script')
parser.add_argument('--i', type=int,  help='this runner is i\'th runner', default=0, required=False)
parser.add_argument('--ofN', type=int, help='all runner number', default=1, required=False)
parser.add_argument('--split', type=str, help='all runner number', default='all', required=True)
parser.add_argument('--checkpoint', type=int, help='all runner number', default=624, required=True)
parser.add_argument('--model', type=str, help='all runner number', default="ARIA", required=True)
parser.add_argument('--subsplit', type=str, help='all runner number', default="test", required=True)
parser.add_argument('--mode', type=str, help='all runner number', default="test_before_finetune", required=True)
args = parser.parse_args()
ALL_RUNNER = args.ofN
THIS_RUNNER = args.i
target_split =  args.split
checkpoint_num = args.checkpoint
target_model = args.model
subsplit = args.subsplit


# target_model = "ARIA"
## current valid: ARIA, UGROUND7B, SEECLICK, AGUVIS, OSATLAS, COGAGENT, QWEN25VL
target_mode = args.mode
## current valid: test_before_finetune, test_after_finetune
# target_split = "android_low"
## only need in test_after_finetune
## current valid: android_low, all
import json
import tqdm
import datasets
import os
import warnings



args = parser.parse_args()
model_path_map = {
    "ARIA":"./aria_ui_model",
    "SEECLICK":"./seeclick_model",
    "UGROUND7B":"./UGround_model_7B",
    "AGUVIS":"./aguvis_model",
    "OSATLAS":"./OSAtlas_model",
    "COGAGENT":"./cogagent_model",
    "QWEN25VL" : ""
}
tuned_model_path_map = {
    "ARIA": f"./output/tuned_aria_ui_model_{target_split}_{checkpoint_num}"
}
os.environ['model_path'] = model_path_map[target_model]
if target_mode == "test_after_finetune":
    os.environ['tuned_model_path'] = tuned_model_path_map[target_model]
os.environ['mode'] = target_mode

my_dataset = datasets.load_dataset("transferability_gui_benchmark", target_split, trust_remote_code=True, cache_dir="./cache")
print(my_dataset)

if target_model == "ARIA":
    if target_mode  == "test_before_finetune":
        from model.aria_ui import run_for_case
    else:
        from model.aria_ui_tuned import run_for_case
elif target_model == "SEECLICK":
    from model.seeclick import run_for_case
elif target_model == "UGROUND7B":
    warnings.warn("Testing UGround7B should involve running the vllm server first as recommended")
    warnings.warn("Example: python -m vllm.entrypoints.openai.api_server --served-model-name osunlp/UGround-V1-7B "
                  "--model ./UGround_model_7B --dtype float16")
    from model.UGround import run_for_case
elif target_model == "AGUVIS":
    from model.aguvis import run_for_case
elif target_model == "OSATLAS":
    from model.osatlas import run_for_case
elif target_model == "COGAGENT":
    from model.cogagent import run_for_case
elif target_model == "QWEN25VL":
    from model.qwen2_5_VL import run_for_case
result = []
if not os.path.exists("./results"):
    os.mkdir("./results")
result_path = f"./results/result_{target_model}_{target_mode}_{target_split}_{checkpoint_num}_{subsplit}_{THIS_RUNNER}.json"
if os.path.exists(result_path):
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
count = 0
current_case_idx = 0
wait_for_run_cases = my_dataset[subsplit]
for case in tqdm.tqdm(wait_for_run_cases):
    if current_case_idx < len(result):
        current_case_idx += 1
        continue
    if current_case_idx % ALL_RUNNER != THIS_RUNNER:
        result.append(None)
        current_case_idx += 1
        continue
    if target_model == "COGAGENT":
        result.append(run_for_case(case['grounding_instruction'], case['screenshot_path'], case['platform_type']))
    else:
        result.append(run_for_case(case['grounding_instruction'], case['screenshot_path']))
    count += 1
    if count % 100 == 0:
        if os.path.exists(result_path):
            shutil.copy(result_path,result_path+".bak")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
    current_case_idx += 1

if os.path.exists(result_path):
    shutil.copy(result_path, result_path + ".bak")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)
