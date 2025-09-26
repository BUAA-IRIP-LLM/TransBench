# 推理脚本使用文档

[BACK](../README.md)

## 环境配置

需求transformers

其余包可视报错情况安装，暂无特定版本需求

## 目录结构及代码运行路径

`model/*.py`负责各模型加载及调用，除`UGROUND7B`外均对外抽象一个run_for_case函数借口，在被外部脚本import时执行模型加载。
`UGROUND7B`遵循推荐的执行方式，采用vllm的方式调用。

* `test.py`为评估主脚本，为方便多机器并行跑，接受参数`--i`, `--ofN`，`--i`为当前机器编号，`--ofN`为机器总数。
* `test.py`会自动加载`model/*.py`中的run_for_case，并调用其run_for_case函数，模型选择由`--model`指定，可以接受的模型名称在脚本中标注。
* `--mode`接受两个模式，分别为`test_before_finetune`,`test_after_finetune`，用于区别测试的模型来源于微调前还是微调后。
* `--split`指定加载的划分名，`--subsplit`选择划分下的测试、训练集，`--checkpoint`用于标记结果输出的文件名，由于仅在训练后才需要区分数据集划分，因此仅对`test_after_finetune`模式有意义。
* 加载的模型的具体位置由 `test.py` 中的字典格式化控制，可修改其中的路径控制加载模型来源，如下：

```python
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
```

通过命令行在所属python环境下运行test脚本后，会加载当前目录的`transferability_gui_benchmark`文件夹保存的数据集，
格式化获取模型路径并设置为环境变量，调用model/*.py完成模型加载，逐个运行run_for_case获取结果后尝试向本地的`./results/***.json`文件输出结果。

> 由于允许多个机器分配任务，对于当前机器没有运行的数据，使用None占位，可据此合并多个机器的结果。

## 结果统计

合并得到完整结果后，可以使用eval_result脚本进行结果分析，该脚本头部手动设置超参数，格式化获得路径后加载分析。

## 可能存在的差异来源

1. 模型推理均采用官方推荐的加载方式以达到峰值性能，因此可能存在temperature导致的随机差异，无法保证每次生成严格一致。
2. 模型训练初始化lora参数时未指定种子，可能导致训练结果受初始化随机性影响。
3. 由于采用自动化流水线配合人工检查生成数据，数据质量仍有提升空间，后续可能会更新数据集版本。



