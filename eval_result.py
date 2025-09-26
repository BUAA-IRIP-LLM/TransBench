target_model = "ARIA"
## current valid: ARIA, UGROUND7B, SEECLICK, AGUVIS, OSATLAS, COGAGENT, QWEN25VL
target_mode = "test_before_finetune"
## current valid: test_before_finetune, test_after_finetune
target_split = "all"
## only need in test_after_finetune
## current valid: android_low, all, normal, web, app
checkpoint_num = 624

subsplit = 'test'

## support transferability_gui_bernchmark[v2]
if target_mode == 'test_before_finetune':
    result_path = f"./results/result_{target_model}_{target_mode}_{target_split}.json"
else:
    result_path = f"./results/result_{target_model}_{target_mode}_{target_split}_{checkpoint_num}_{subsplit}.json"
import json
import re

import datasets

my_dataset = datasets.load_dataset("transferability_gui_benchmark", target_split, trust_remote_code=True,
                                   cache_dir='./cache')
print(my_dataset)


def extract_coordinates(click_command):
    # 正则表达式模式用于匹配x和y的值
    pattern = r'.*pyautogui\.click\(.*x=(.*?),.*y=(.*?)\).*'

    match = re.search(pattern, click_command)
    if match:
        # 如果找到匹配项，则提取x和y的值，并尝试将它们转换为整数
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            return x, y
        except:
            print("error:{}".format(click_command))
            return -1, -1

    else:
        print("error:{}".format(click_command))
        print("error")
        return -1, -1


def extract_coordinates_os_atlas(click_command):
    pattern = r'.*<\|box_start\|>(.*)<\|box_end\|>.*'
    match = re.search(pattern, click_command)
    if match:
        # 如果找到匹配项，则提取x和y的值，并尝试将它们转换为整数
        try:
            bbox = match.group(1)
            bbox = (bbox.replace("]", " ").replace("[", ' ')
                    .replace(",", ' ').replace("(", ' ').replace(")", ' '))
            bbox = bbox.split()
            if len(bbox) != 4:
                raise ValueError("bbox error")
            bbox = [int(x) for x in bbox]
            bbox = [x / 1000.0 for x in bbox]
            x, y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            return x, y
        except:
            print("error:{}".format(click_command))
            return -1, -1

    else:
        print("error:{}".format(click_command))
        print("error")
        return -1, -1


count = 0
with open(result_path, 'r', encoding='utf-8') as f:
    results = json.load(f)
if len(results) != len(my_dataset[subsplit]):
    print(f"result len:{len(results)}, dataset len:{len(my_dataset[subsplit])}")
    raise ValueError("result length error")


def calculate_distance(bbox, point)-> float:
    if point[0]< 0 or point[1] < 0 or point[0] > 1 or point[1] > 1:
        return 0.5*(2**0.5)
    def eu_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    center_of_box = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return eu_distance(point, center_of_box)
    # bbox (x1,y1,x2,y2)
    # bbox (0, 1, 2, 3)
    x1,y1,x2,y2 = bbox
    box = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
    idx = [(0,1), (2,1), (0,3), (2,3)]
    corners = [[bbox[j] for j in idx[i]] for i in range(4)]
    if box[0] < point[0] < box[2] and box[1] < point[1] < box[3]:
        return 0

    if box[0] < point[0] < box[2]:
        # 计算到上下边的最短距离
        distance_to_top = abs(point[1] - box[1])
        distance_to_bottom = abs(point[1] - box[3])
        return min(distance_to_top, distance_to_bottom)

    if box[1] < point[1] < box[3]:
        # 计算到左右边的最短距离
        distance_to_left = abs(point[0] - box[0])
        distance_to_right = abs(point[0] - box[2])
        return min(distance_to_left, distance_to_right)
    corner_min = min([eu_distance(point, corners[i]) for i in range(4)])
    return corner_min
def evaluate_result_standard():
    correct_count = 0
    error_count = 0
    platform_count = {}
    error_cases = []
    error_points = []
    error_resuts = []
    for i in range(len(results)):
        result = results[i]
        if target_model == "AGUVIS":
            result = extract_coordinates(result)
        elif target_model == "OSATLAS":
            result = extract_coordinates_os_atlas(result)
        elif target_model == "COGAGENT":
            while type(result[0]) is list:
                result = result[0]
            if not len(result) == 4:
                result = [-1, -1, -1, -1]
            result = [(result[0] + result[2]) / 2, (result[1] + result[3]) / 2]
        elif target_model == "QWEN25VL":
            if 'arguments' not in result or 'action' not in result['arguments'] or 'coordinate' not in result['arguments']:
                result = [-1, -1, -1, -1]
            else:
                _w, _h = result['origin_size']
                result = result['arguments']['coordinate']
                try:
                    result = [result[0]/_w, result[1]/_h]
                except:
                    result = [-1, -1]
        correct_result = my_dataset[subsplit][i]['target_bbox'][0]
        platform = my_dataset[subsplit][i]['platform_type']
        version_type = my_dataset[subsplit][i]['app_version_type']
        screenshot_path = my_dataset[subsplit][i]['screenshot_path']

        key_name = platform + "-" + version_type
        if key_name not in platform_count:
            platform_count[key_name] = {'correct': 0, 'error': 0, "error_distance":0.0}
        result = [x for x in result]
        distance = calculate_distance(correct_result, result)
        platform_count[key_name]['error_distance'] += distance
        if correct_result[0] < result[0] < correct_result[2] and correct_result[1] < result[1] < correct_result[3]:
            correct_count += 1
            platform_count[key_name]['correct'] += 1
            error_resuts.append({
                "result": result,
                "correct_result": correct_result,
                "platform": platform,
                "version_type": version_type,
                "screenshot_path": screenshot_path.replace("\\", '/').split("/")[-1],
                "grounding_instruction": my_dataset[subsplit][i]['grounding_instruction'],
                "cate": my_dataset[subsplit][i]['platform_type'],
                "is_right": True
            })
        else:
            error_resuts.append({
                "result": result,
                "correct_result": correct_result,
                "platform": platform,
                "version_type": version_type,
                "screenshot_path": screenshot_path.replace("\\", '/').split("/")[-1],
                "grounding_instruction": my_dataset[subsplit][i]['grounding_instruction'],
                "cate": my_dataset[subsplit][i]['platform_type'],
                "is_right": False
            })
            error_count += 1
            platform_count[key_name]['error'] += 1
            error_cases.append(my_dataset[subsplit][i])
            error_points.append(result)
    with open("./error_results.json", 'w') as f:
        json.dump(error_resuts, f, indent=4)
    print(correct_count / (correct_count + error_count))
    print(correct_count, error_count)
    print("accuracy")
    all_correct = 0
    all_error = 0
    all_distance = 0
    for key_name in platform_count:
        print(
            f"{key_name}\t{platform_count[key_name]['correct'] / (platform_count[key_name]['correct'] + platform_count[key_name]['error'])}",
            end="\t")
        all_correct += platform_count[key_name]['correct']
        all_error += platform_count[key_name]['error']
        all_distance += platform_count[key_name]['error_distance']
    # print("err")
    # for key_name in platform_count:
    #     print(
    #         f"{key_name}\t{100*platform_count[key_name]['error_distance'] / (platform_count[key_name]['error'])}",
    #         end="\t")
    print("average distance")
    for key_name in platform_count:
        print(
            f"{key_name}\t{100*platform_count[key_name]['error_distance'] / (platform_count[key_name]['correct'] + platform_count[key_name]['error'])}",
            end="\t")


    print("latex_output")
    android_accuracy = ((platform_count['android-low']['correct']+platform_count['android-high']['correct'])/
                        (platform_count['android-low']['correct']+platform_count['android-high']['correct']+platform_count['android-low']['error']+platform_count['android-high']['error']))
    android_distance = ((platform_count['android-low']['error_distance']+platform_count['android-high']['error_distance'])/
                        (platform_count['android-low']['correct']+platform_count['android-high']['correct']+platform_count['android-low']['error']+platform_count['android-high']['error']))
    all_accuracy = (all_correct / (all_correct + all_error))
    all_average_distance = (all_distance / (all_correct + all_error))
    # print("{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(all_accuracy * 100, all_average_distance * 100, android_accuracy * 100,
    #                                                  android_distance * 100), end=" & ")
    print("{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(all_accuracy * 100, all_average_distance * 100, android_accuracy * 100,
                                                     android_distance * 100), end=" & ")

    for key_name in platform_count:
        print(
            # "\\textbf{"+
            "{:.2f}".format(100*platform_count[key_name]['correct'] / (platform_count[key_name]['correct'] + platform_count[key_name]['error']))
              # +"}"
        ,end=" & ")

        print(
            # "\\textbf{"+
            "{:.2f}".format(100*platform_count[key_name]['error_distance'] / (platform_count[key_name]['correct'] + platform_count[key_name]['error']))
            # +"}"
        ,end=" & ")
def evaluate_result_app_level():
    choosed_app = set([])
    choosed_type = set([])
    for case in my_dataset['train']:
        choosed_app.add(case['app_name'])
        choosed_type.add(case['app_type'])
    standard_dataset = datasets.load_dataset("transferability_gui_benchmark", 'all', trust_remote_code=True,
                                           cache_dir='./cache')
    with open("./results/result_ARIA_test_before_finetune_all_0_test.json", 'r', encoding='utf-8') as f:
        snandard_results = json.load(f)
    def calculate_3type_result(this_results, this_dataset):
        same_type_platform_count = {}
        out_type_platform_count = {}
        correct_count = 0
        error_count = 0
        platform_count = {}
        error_cases = []
        error_points = []
        for i in range(len(this_results)):
            result = this_results[i]
            if target_model == "AGUVIS":
                result = extract_coordinates(result)
            elif target_model == "OSATLAS":
                result = extract_coordinates_os_atlas(result)
            elif target_model == "COGAGENT":
                while type(result[0]) is list:
                    result = result[0]
                if not len(result) == 4:
                    result = [-1, -1, -1, -1]
                result = [(result[0] + result[2]) / 2, (result[1] + result[3]) / 2]
            elif target_model == "QWEN25VL":
                if 'arguments' not in result or 'action' not in result['arguments'] or 'coordinate' not in result['arguments']:
                    result = [-1, -1, -1, -1]
                else:
                    _w, _h = result['origin_size']
                    result = result['arguments']['coordinate']
                    try:
                        result = [result[0]/_w, result[1]/_h]
                    except:
                        result = [-1, -1]
            if result[0] > 1:
                result = [result[0]/1000, result[1]/1000]
            correct_result = this_dataset[subsplit][i]['target_bbox'][0]
            platform = this_dataset[subsplit][i]['platform_type']
            version_type = this_dataset[subsplit][i]['app_version_type']
            app_name = this_dataset[subsplit][i]['app_name']
            if app_name in choosed_app:
                continue
            app_type = this_dataset[subsplit][i]['app_type']
            key_name = platform + "-" + version_type
            if key_name not in platform_count:
                platform_count[key_name] = {'correct': 0, 'error': 0, "error_distance":0.0}
                same_type_platform_count[key_name] = {'correct': 0, 'error': 0, "error_distance":0.0}
                out_type_platform_count[key_name] = {'correct': 0, 'error': 0, "error_distance":0.0}
            result = [x for x in result]
            distance = calculate_distance(correct_result, result)
            platform_count[key_name]['error_distance'] += distance
            if app_type in choosed_type:
                same_type_platform_count[key_name]['error_distance'] += distance
            else:
                out_type_platform_count[key_name]['error_distance'] += distance
            if correct_result[0] < result[0] < correct_result[2] and correct_result[1] < result[1] < correct_result[3]:
                correct_count += 1
                platform_count[key_name]['correct'] += 1
                if app_type in choosed_type:
                    same_type_platform_count[key_name]['correct'] += 1
                else:
                    out_type_platform_count[key_name]['correct'] += 1
            else:
                error_count += 1
                platform_count[key_name]['error'] += 1
                error_cases.append(this_dataset[subsplit][i])
                error_points.append(result)
                if app_type in choosed_type:
                    same_type_platform_count[key_name]['error'] += 1
                else:
                    out_type_platform_count[key_name]['error'] += 1
        print(correct_count / (correct_count + error_count))
        print(correct_count, error_count)
        return platform_count, same_type_platform_count, out_type_platform_count

    def _evaluate(platform_count, same_type_platform_count, out_type_platform_count, special_str):
        print(special_str)
        print("=================")
        print("outputing overall, sametype, othertype app performance")
        result = []
        for target_count in [platform_count, same_type_platform_count, out_type_platform_count]:
            result.append([])
            print("\naccuracy")
            all_correct = 0
            all_error = 0
            all_distance = 0
            for key_name in target_count:
                val = target_count[key_name]['correct'] / (target_count[key_name]['correct'] + target_count[key_name]['error'])
                result[-1].append(val)
                print(
                    f"{key_name}\t{target_count[key_name]['correct'] / (target_count[key_name]['correct'] + target_count[key_name]['error'])}",
                    end="\t")
                all_correct += target_count[key_name]['correct']
                all_error += target_count[key_name]['error']
            result.append([])
            print("\naverage distance")
            for key_name in target_count:
                all_distance += target_count[key_name]['error_distance']
                val = 100 * target_count[key_name]['error_distance'] / (
                            target_count[key_name]['correct'] + target_count[key_name]['error'])
                result[-1].append(val)
                print(
                    f"{key_name}\t{100*target_count[key_name]['error_distance'] / (target_count[key_name]['correct'] + target_count[key_name]['error'])}",
                    end="\t")
            print(f"\nall accuracy and distance:{all_correct/(all_correct+all_error)}, {100*all_distance/(all_correct+all_error)}" )
        return result
    evares1= _evaluate(*(res1:=calculate_3type_result(snandard_results, standard_dataset)), special_str="standard")
    evares2 = _evaluate(*(res2:=calculate_3type_result(results, my_dataset)), special_str="ours")
    isacc = True
    for i in range(len(evares2)):
        for j in range(len(evares2[i])):
            evares2[i][j] = (evares2[i][j] - evares1[i][j])
        isacc = not isacc
    print(evares2)
    def format_to_latex(lists):
        print("")
        for line in lists:
            for item in line:
                print(f" & {item:.2f}", end="\t")
            print("\\\\ \\hline")
    format_to_latex(evares2)



if __name__ == '__main__':
    if target_split=='app':
        evaluate_result_app_level()
    else:
        evaluate_result_standard()