# TransBench

[![ACL Findings Accepted Badge](https://img.shields.io/badge/ACL%20Findings-2025-4d71a3)](https://2025.aclweb.org/)

This repository contains the code and data for our paper **"TransBench: Breaking Barriers for Transferable Graphical User
 Interface Agents in Dynamic Digital Environments**, which has been accepted to [ACL Findings 2025](https://2025.aclweb.org/).

## 📢 News
- **May 2025**: Our paper has been accepted to ACL Findings 2025!
- **September 2025**: The Benchmark and test scripts are released on [huggingface datasets](https://huggingface.co/datasets/luyuheng/TransBench/tree/main). 

## 📝 Paper Abstract
Graphical User Interface (GUI) agents, which autonomously operate on digital interfaces through natural language instructions, 
hold transformative potential for accessibility, automation, and user experience. 
A critical aspect of their functionality is grounding — the ability to map linguistic intents to visual and structural interface elements. 
However, existing GUI agents often struggle to adapt to the dynamic and interconnected nature of real-world digital environments, 
where tasks frequently span multiple platforms and applications while also being impacted by version updates. 
To address this, we introduce TransBench, 
the first benchmark designed to systematically evaluate and enhance the transferability of GUI agents across three key dimensions: 
crossversion transferability (adapting to version updates), cross-platform transferability (generalizing across platforms like iOS, Android, and Web), 
and cross-application transferability (handling tasks spanning functionally distinct apps). 
TransBench includes 15 app categories with diverse functionalities, capturing essential pages across versions and platforms to enable robust evaluation. 
Our experiments demonstrate significant improvements in grounding accuracy, showcasing the practical utility of GUI agents in dynamic, realworld environments. 

## 🚀 Features
- Includes **80 real-world applications across 15 categories** representing diverse usage scenarios.
- Most applications provide **three platform variants** (iOS/Android/Web), with **Android offering two App versions**.

## 📚 Documents

Test scripts document: [chinese document](./docs/infer_doc.md)

## ⚠️ Limitations

Since this benchmark employs an automated pipeline for example generation, despite our best efforts to ensure data quality, 
the dataset still contains certain errors—such as inconsistent descriptions of object positions or shapes, non-unique answers, etc.
—due to cost constraints and the limited capabilities of vision-language (VL) models available at the time of the paper's release. 

While these imperfections are acceptable for evaluation and supervised fine-tuning (SFT), they should be taken into account when using the dataset for research methods sensitive to erroneous examples, such as GRPO.

## 📄 Citation
If you use our work, please cite:
```bibtex
@inproceedings{yourcitationkey,
  title={TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments},
  author={Lu, Yuheng and Yu, Qian and Wang, Hongru and Liu, Zeming and Su, Wei and Liu, Yanping and Guo, Yuhang and Liang, Maocheng and Wang, Yunhong and Wang, Haifeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
