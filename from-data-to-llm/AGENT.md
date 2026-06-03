# from-data-to-llm 协作说明

本文面向协作者和自动化 agent，用于快速准备 `from-data-to-llm` 教程运行环境。

## 项目结构

- `README.md`：教程首页说明，包含课程简介和交互课程图。
- `docs/index.md`：迁移到 LazyLLM 文档后的中文教程入口页。
- `docs/outline.md`：完整课程大纲，包含各部分目标和 28 个课时的详细内容。
- `docs/assets/course_map.html`：交互式课程图，被首页和大纲页通过 iframe 引用。
- `docs/chapter*/`：各课时正文、代码、图片、视频等教学材料。
- `site/`：本地构建产物目录，不应作为教程源文件提交。

## 课程目录摘要

这部分用于帮助协作者和 agent 快速理解 28 讲内容定位。更详细的分章节目标与知识点，
请以 `docs/outline.md` 为准。

- 第1讲 Transformer 核心与 Self-Attention 深度剖析
  从注意力机制、位置编码到 Transformer 结构，理解大模型架构的基础。
- 第2讲 LLM 训练范式与数据工程全景
  建立从数据准备、训练流程到评测反馈的大模型训练全局视角。
- 第3讲 分布式训练技术概览
  讲解数据并行、模型并行、流水线并行等大规模训练核心技术。
- 第4讲 模型部署与推理加速
  介绍推理部署、显存优化、服务化与加速策略。
- 第5讲 基于 Agent 的数据处理
  使用 Agent 自动化完成数据清洗、分析、生成与流程编排。
- 第6讲 基于 LazyLLM 的全流程实践
  以 LazyLLM 串联数据构建、模型训练与效果评测。
- 第7讲 预训练原理、策略与评测
  理解预训练目标、训练策略和衡量模型能力的关键指标。
- 第8讲 预训练数据构建全流程
  覆盖语料采集、清洗、去重、过滤、配比与数据质量控制。
- 第9讲 基于 LazyLLM 的预训练实战
  通过实践完成预训练数据到模型训练的闭环。
- 第10讲 多模态 LLM 架构与预训练实战
  解析视觉编码、图文对齐与多模态预训练方法。
- 第11讲 指令微调原理与策略
  讲解 SFT、LoRA 等指令微调方法及其数据要求。
- 第12讲 通用指令数据构建、合成与蒸馏
  构建高质量指令数据，并利用合成与蒸馏提升覆盖度。
- 第13讲 基于 LazyLLM 的微调实战
  使用 LazyLLM 完成微调任务配置、训练和效果验证。
- 第14讲 多模态指令微调与实战
  面向图文任务构建多模态指令数据并完成微调实践。
- 第15讲 对齐算法原理（RLHF & GRPO）
  理解偏好优化、强化学习对齐与 GRPO 等主流对齐算法。
- 第16讲 偏好数据构建
  讲解偏好样本设计、标注、质量控制和对齐训练数据组织。
- 第17讲 模型风险、合规与伦理
  识别大模型安全风险，理解合规、隐私和伦理治理要求。
- 第18讲 基于 LazyLLM 的对齐实战
  结合 LazyLLM 完成对齐数据、训练流程和结果评估实践。
- 第19讲 推理与数学能力增强
  围绕推理链、数学数据和训练策略提升模型复杂问题求解能力。
- 第20讲 代码能力增强
  构建代码语料和训练任务，增强模型代码理解与生成能力。
- 第21讲 长上下文能力增强
  介绍长上下文数据构建、位置扩展与长序列评测方法。
- 第22讲 结构化输出与格式对齐
  让模型稳定输出 JSON、表格、工具参数等可解析结构。
- 第23讲 Agent 能力增强（Tools & Planning）
  通过工具调用、任务规划和执行反馈增强 Agent 能力。
- 第24讲 行业领域模型实战
  面向垂直行业构建领域数据、训练方案和落地评测流程。
- 第25讲 RAG 架构原理与数据处理
  讲解 RAG 系统结构、知识处理、切分、索引与检索数据准备。
- 第26讲 Embedding 模型微调与实战
  构建检索训练样本并微调 Embedding 模型，提升召回质量。
- 第27讲 Reranker 模型微调与实战
  通过重排模型微调提升候选文档排序与最终问答效果。
- 第28讲 Agentic RAG 能力增强
  融合 Agent 与 RAG，实现多跳检索、规划执行和复杂任务增强。

## 环境配置

模型训练依赖 Python 3.10 环境，建议使用 conda 创建独立环境：

```bash
conda create -n from-data-to-llm python=3.10
conda activate from-data-to-llm
```

## 数据工程依赖

```text
gymnasium==1.2.3
trafilatura==2.0.0
math_verify==0.9.0
datasketch==1.9.0
z3-solver==4.16.0.0
wikipedia==1.4.0
sympy==1.14.0
```

## 模型训练依赖

```text
transformers==4.57.1
torch==2.7.1
datasets==2.21.0
pillow==11.3.0
tqdm==4.67.1
numpy==1.26.4
json5==0.9.25
regex==2024.7.24
requests==2.32.3
huggingface-hub==0.35.3
matplotlib==3.9.2
modelscope==1.27.1
json_repair==0.57.1
jieba==0.42.1
setuptools<71
sentence_transformers
```

## 工具类依赖

```text
pypdf
docx2txt
ebooklib
html2text
olefile
openpyxl
python-pptx
tiktoken
spacy
bm25s
pystemmer
nltk
sentencepiece
psycopg2-binary
sqlalchemy
```
