# LLM 模型训练与数据工程 (28课时)

## 课程简介
本课程旨在为开发者和算法工程师提供一套**从数据到模型再到应用**的 LLM 全栈实战指南。课程核心聚焦于**数据工程**，并将其深度融入到模型训练的每一个环节——从底座预训练 (Pre-training) 到指令微调 (SFT)，再到人类价值观对齐 (RLHF/GRPO)。

课程不仅涵盖了纯文本、多模态、Embedding 等多维度的技术原理，更引入了**系统工程**视角，详解分布式训练、高效部署与模型合规。特别值得一提的是，本课程贯穿了 **LazyLLM** 全流程实战与 **Agent（智能体）** 的双重应用：既教授如何构建具备 Agent 能力的模型，也演示如何利用 Agent 自动化流水线来清洗和合成高质量数据，助力企业构建闭环的“数据飞轮”。

<iframe src="../assets/course_map.html" width="100%" style="border:none; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); min-height: 600px;" onload="this.style.height = this.contentWindow.document.documentElement.scrollHeight + 'px'" scrolling="no"></iframe>

## 课程大纲

### 第一部分：大模型基础：架构、训练与数据范式 (2课时)
**目标**：深入剖析Transformer架构核心，确立“模型-训练-数据”三位一体的工程认知。

-   **第1课时：Transformer 核心与 Self-Attention 深度剖析**
    -   Transformer 架构回顾：Encoder-Decoder vs Decoder-only。
    -   **核心机制详解**：Self-Attention 计算原理 (Q/K/V)，Multi-head 机制，Masking 策略。
    -   位置编码 (Positional Encoding)：RoPE, ALiBi 核心原理与外推性 (Extrapolation) 理论分析。
    -   计算复杂度分析：$O(N^2)$ 问题与 KV Cache，理解数据长度对训练资源的约束。
    -   主流模型架构演进 (BERT, T5, GPT-3, Llama, Mistral, MoE, DeepSeek, Qwen3)。
-   **第2课时：LLM 训练范式与数据工程全景**
    -   **训练范式**：Pre-training -> SFT -> RLHF/PPO/DPO/GRPO 全流程解析。
    -   **数据工程范式**：Data-Centric AI 理念，从“以模型为中心”到“以数据为中心”。
    -   Chinchilla法则
    -   数据生命周期管理：采集 -> 清洗 -> 过滤 -> 扩展/合成 -> 去重 -> 评估 -> 配比 -> 课程学习。

### 第二部分：系统工程基础：分布式、部署与Agent工具 (4课时)
**目标**：在处理数据前，先理解大规模训练系统的底座，掌握模型部署核心技术，并将Agent作为数据处理的基础工具，最后通过LazyLLM实践全流程。

-   **第3课时：分布式训练技术概览**
    -   **通信原语与硬件基础**：AllReduce, AllGather, ReduceScatter 原理，NVLink/InfiniBand 网络拓扑对训练的影响。
    -   **并行策略详解**：
        -   数据并行：DDP vs FSDP (Fully Sharded Data Parallel) 原理对比。
        -   模型并行：Tensor Parallelism (TP) 与 Pipeline Parallelism (PP) 的切分逻辑。
        -   混合并行：3D 并行 (DP+TP+PP) 与 MoE 专家并行 (EP) 的组合策略。
    -   **显存优化技术**：ZeRO (1/2/3) 系列，FlashAttention-2/3 加速原理，Gradient Checkpointing (重计算)，CPU Offload。
    -   **分布式对数据的影响**：Global Batch Size 缩放规律，Micro Batch 流水线填充，数据分片 (Sharding) 与断点续训 (Checkpointing) 策略。
-   **第4课时：模型部署与推理加速**
    -   推理引擎架构：vLLM (PagedAttention) 核心原理。
    -   吞吐优化技术：Continuous Batching, Speculative Decoding (投机采样/Medusa), KV Cache Quantization。
    -   部署中的量化策略：AWQ, GPTQ, SmoothQuant 及其对模型精度的影响。
    - **实践**：LazyLLM 的基础使用，部署一个LLM推理服务。
-   **第5课时：Agent-Based Data Processing (基于Agent的数据处理)**
    -   Agent 概述：定义、类型 (单Agent vs Multi-Agent)、核心能力 (工具调用、规划与决策)。
    -   Agent 策略：function-call, react, plan-and-solve, rewoo 等主流方法解析。
    -   **Agent 作为数据工人**：利用 Multi-Agent 系统进行数据清洗、标注和审查。
    -   **自动化流水线**：构建 Agent Workflow 自动生成合成数据并进行自我验证 (Self-Correction)。
    -   **实践**：LazyLLM 的基础使用，与 Agent 结合搭建一个自动化的数据质量优化系统。
-   **第6课时：基于 LazyLLM 的数据-训练-推理全流程实践**
    -   LazyLLM 框架解析：低代码构建高效的 LLM 迭代闭环。
    -   **数据流 (Data Flow)**：自动化数据筛选、扩展与合成策略。
    -   **模型流 (Model Flow)**：从训练 (Training) 到服务化 (Serving) 的无缝衔接。
    -   **实践**：使用 LazyLLM 搭建一个“数据清洗 -> 模型训练 -> 应用部署”的一条龙流水线。

### 第三部分：预训练数据工程与实战 (Pre-training) (4课时)
**目标**：遵循“原理-数据-实战”逻辑，掌握从底座模型训练原理到海量数据构建的全流程技术。

-   **第7课时：预训练原理、策略与评测**
    -   **核心原理**：Next Token Prediction 范式，因果掩码 (Causal Masking)，MoE 路由机制。
    -   **训练策略调优**：学习率调度 (Cosine Decay/WSD), 优化器选择 (AdamW/Lion), 梯度裁剪。
    -   **模型评测**：Perplexity (PPL) 指标，MMLU/GSM8K/HumanEval 等下游任务评测基准。
-   **第8课时：预训练数据构建全流程**
    -   **文本数据集构建**：CommonCrawl/GitHub 等数据源解析，PII 移除与启发式过滤 (Heuristic Filtering)。
    -   **去重 (Deduplication)**：MinHash + LSH 模糊去重，Exact 去重及其对 Scaling Laws 的影响。
    -   **质量评估 (Quality Eval)**：基于统计规则 vs 基于模型打分 (Quality Classifier) 的数据分级体系。
    -   **Tokenizer**：BPE/Unigram 原理，词表扩充 (Vocabulary Expansion) 对多语言能力的影响。
    -   **配比与课程**：数据混合 (Data Mixing) 策略，退火阶段 (Annealing) 的高质量数据配比。
-   **第9课时：基于 LazyLLM 的预训练全链路实战**
    -   **数据准备**：使用 LazyLLM 进行大规模文本数据的清洗、去重与 Tokenization。
    -   **训练启动**：配置分布式预训练任务，监控 Loss 曲线。
    -   **推理与评测**：训练后模型的自动化评测与简单的推理服务搭建。
-   **第10课时：多模态 LLM 架构与预训练实战**
    -   Vision Encoder + LLM 架构 (如 LLaVA, Qwen-VL)。
    -   **图文数据集构建**：Image-Text Pairs 处理 (LAION/COYO)，分辨率调整与 Patch 切分。
    -   **LazyLLM 实战**：构建多模态预训练流水线 (数据准备 -> 模型组装 -> 训练启动)。

### 第四部分：微调数据工程与实战 (SFT) (4课时)
**目标**：遵循“原理-数据-实战”逻辑，掌握指令微调的核心原理、通用数据构建及多模态微调实战。

-   **第11课时：指令微调 (Instruction Tuning) 原理与策略**
    -   **核心原理**：从续写到对话——指令遵循能力的激发机制与对齐税 (Alignment Tax)。
    -   **参数高效微调 (PEFT)**：LoRA, QLoRA, AdaLoRA, DoRA 原理详解。
    -   **全量微调 (Full Fine-tuning)**：适用场景与显存优化 (ZeRO-3 Offload)。
    -   **防止过拟合**：NEFTune (噪声嵌入) 与 Pack 训练策略。
-   **第12课时：通用指令数据构建、合成与蒸馏**
    -   **指令数据集构建**：System Prompt, User, Assistant (多轮对话格式) 的标准化构建。
    -   **数据蒸馏 (Data Distillation)**：利用强模型 (Teacher) 生成高质量合成数据 (Synthetic Data) 的方法论。
    -   **基于 RAG 的数据合成**：利用检索增强生成构建基于文档的高质量问答对 (QA Pairs)。
    -   **数据进化与增强**：Evol-Instruct 提升复杂度，多样性 (Diversity) 扩充与风格迁移。
    -   **SFT 数据质量评估**：基于 IFD (Instruction Following Difficulty) 指标与模型打分的筛选策略。
-   **第13课时：基于 LazyLLM 的微调全链路实战**
    -   **数据准备**：构造高质量指令微调数据集 (Alpaca/ShareGPT 格式)。
    -   **微调启动**：配置 LoRA 参数，启动单机多卡/多机多卡微调。
    -   **效果验证**：使用推理模式进行人工评估与自动评测。
-   **第14课时：多模态指令微调与实战**
    -   Visual Instruction Tuning：从 Caption 到 VQA (Visual Question Answering)。
    -   **多模态指令数据集**：构造多模态 CoT 数据，交错图文 (Interleaved Image-Text) 数据处理。
    -   **LazyLLM 实战**：配置 LoRA 参数适配多模态模块，进行图文交互测试。

### 第五部分：对齐数据工程与实战 (Alignment) (4课时)
**目标**：遵循“原理-数据-实战”逻辑，掌握RLHF/GRPO核心算法、偏好数据构建、安全合规及对齐实战。

-   **第15课时：对齐算法原理 (RLHF & GRPO)**
    -   **RLHF 基础**：Reward Model (RM) 与 PPO (Proximal Policy Optimization) 算法详解。
    -   **直接偏好优化**：DPO (Direct Preference Optimization), IPO, KTO 原理对比。
    -   **前沿技术**：DeepSeek-R1 中的 GRPO (Group Relative Policy Optimization) 与强化推理能力。
-   **第16课时：偏好数据 (Preference Data) 构建**
    -   **偏好数据集构建**：构造 Chosen vs Rejected (Pairwise / Listwise) 数据格式。
    -   标注方法：人工排序 vs 模型打分 (LLM-as-a-Judge)。
    -   **过程监督数据集**：Process Reward Model (PRM) 的 Step-by-step 验证数据构建 (Math-Shepherd)。
    -   **规则奖励数据集**：基于答案正确性与格式合规性的 Rule-based Reward 数据。
-   **第17课时：模型风险、合规与伦理 (Risk, Compliance & Ethics)**
    -   **风险图谱**：幻觉 (Hallucination)、偏见 (Bias)、毒性 (Toxicity) 与越狱 (Jailbreak) 防御。
    -   **红队测试 (Red Teaming)**：自动化攻击提示词构建 (Attack Prompts) 与对抗样本生成。
    -   **合规与伦理**：数据版权审查、隐私合规 (GDPR/CCPA) 与负责任的 AI (Responsible AI) 原则。
-   **第18课时：基于 LazyLLM 的对齐全链路实战**
    -   **RM 训练**：使用偏好数据训练奖励模型。
    -   **RL 训练**：配置 PPO/GRPO 参数，启动强化学习训练。
    -   **效果评估**：对比对齐前后的模型在安全性与指令遵循上的表现。


### 第六部分：特定领域能力数据构建 (Domain-Specific Data) (6课时)
**目标**：深入垂直领域，掌握代码、数学、Agent等高难度数据的构建，以及行业模型的全流程训练。

-   **第19课时：推理 (Reasoning)与数学能力增强**
    -   **推理数据集 (Reasoning Data)**：构造 Chain-of-Thought (CoT) 推理路径，合成复杂逻辑数据。
    -   **数学数据集 (Math Data)**：增加推理步骤 (Step-by-step), 格式化数学公式 (LaTeX), 过程验证 (Process Verification)。
    -   验证驱动的数据过滤 (利用解释器/求解器验证数据正确性)。
-   **第20课时：代码能力增强**
    -   **代码数据集构建**：GitHub 仓库抓取策略，依赖解析与文件拓扑排序 (Topological Sorting) 以保持上下文逻辑。
    -   **预训练策略**：Fill-in-the-Middle (FIM) 任务设计及其对代码补全能力的影响。
    -   **指令微调**：构造代码数据生成，代码生成与单元测试生成 (Unit Test Generation) 数据。
    -   **执行反馈 (Execution Feedback)**：构建基于编译器/解释器反馈的强化学习环境 (Code RL)，利用测试通过率作为 Reward。
-   **第21课时：长上下文能力增强**
    -   **长文本数据构建**：书籍/论文/财报的长文本拼接，跨文档上下文关联保留策略。
    -   **合成长数据**：通过 "Needle In A Haystack" (大海捞针) 任务合成针对性训练数据，提升长窗口下的检索准确率。
    -   **LazyLLM实战与评测**：大海捞针测试 (NIAH), LongBench 评测集与困惑度 (PPL) 的长距离衰减监控。
-   **第22课时：结构化输出与格式对齐**
    -   **数据构建流水线**：Schema 设计 (JSON/Pydantic) -> 逆向合成 (基于 Schema 生成 JSON 再反推文本) -> 自动化校验与清洗 -> 负样本构建。
    -   **关键技术**：TypeScript 风格提示工程，语法引导解码 (Grammar-guided Decoding) 原理与基于 Trie 树的推理约束。
    -   **评测指标**：格式错误率、字段级准确率与幻觉率。
    -   **LazyLLM 实战**：训练一个结构化信息抽取模型，完成从数据准备、SFT 微调到能够稳定输出JSON 的全流程。
-   **第23课时：Agent 能力增强 (Tools & Planning)**
    -   **工具调用数据集 (Tool Use Data)**：API 定义、参数生成、调用轨迹 (Trace) 数据构建。
    -   **规划能力数据集 (Planning Data)**：合成 ReAct, Plan-and-Solve 等模式的思考-行动轨迹。
    -   多轮对话中的状态保持与环境反馈模拟数据。
-   **第24课时：行业领域模型实战 (Industry Domain Training)**
    -   **行业数据集准备**：垂直领域（如医疗、法律、金融）的数据清洗、脱敏与知识图谱融合。
    -   **继续预训练 (CPT)**：领域知识注入的训练策略与数据配比。
    -   **领域指令数据集**：构建符合行业业务逻辑的 SFT 指令集。
    -   **LazyLLM 实战**：搭建“行业语料 CPT -> 业务指令 SFT”的完整训练流水线。


### 第七部分：检索增强生成 (RAG) 数据工程 (4课时)
**目标**：理解 RAG 核心架构，掌握从文档处理到 Embedding/Reranker 模型微调的全流程数据工程。

-   **第25课时：RAG 架构原理与数据处理**
    -   **核心范式**：RAG 解决幻觉与时效性问题，RAG vs Long Context 优劣分析。
    -   **架构拆解**：Retrieval (检索), Augmentation (增强), Generation (生成) 全流程。
    -   **LazyLLM 实战**：利用 Agent 进行文档的智能解析、摘要生成与元数据增强 (Metadata Enrichment) 入库。
    -   **微调必要性**：通用模型在特定领域的局限性，引出 Embedding 与 Reranker 微调的价值。
-   **第26课时：Embedding 模型微调与实战**
    -   **核心原理**：Bi-encoder 架构，对比学习 (Contrastive Learning) 损失函数 (InfoNCE)。
    -   **Embedding 数据集构建**：正负样本对挖掘，难负样本 (Hard Negatives) 的重要性与挖掘策略。
    -   **LazyLLM 实战**：构建文本对数据，微调 Embedding 模型并评估 MTEB 指标。
-   **第27课时：Reranker 模型微调与实战**
    -   **核心原理**：Cross-encoder 架构，相关性打分机制与计算开销分析。
    -   **Reranker 数据集构建**：利用 LLM 蒸馏生成排序数据，Listwise vs Pairwise 数据格式。
    -   **LazyLLM 实战**：训练 Reranker 模型，并搭建 RAG 流水线对比检索效果。
-   **第28课时：Agentic RAG 能力增强**
    -   **从 RAG 到 Agentic RAG**：引入 Planning 与 Reflection 机制解决复杂多跳问题 (Multi-hop QA)。
    -   **数据构建**：构造 Self-RAG (自省式 RAG) 数据，包含检索意图识别、文档相关性反思 (IsRel) 与回复生成质量打分 (IsSup)。
    -   **工具化检索**：将搜索引擎/向量库封装为 Tool，训练模型自主决定 "何时检索" (Adaptive Retrieval) 及 "如何改写查询" (Query Rewriting)。
    -   **Graph RAG**：利用知识图谱增强检索数据，捕捉跨文档的实体关系与全局摘要能力。


### 课程总结
-   **构建企业级 LLM 数据飞轮 (Data Flywheel)** —— 数据闭环的重要性。
