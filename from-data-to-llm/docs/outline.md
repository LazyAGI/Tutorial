# LLM 模型训练与数据工程 (28课时)

## 课程简介
本课程旨在为开发者和算法工程师提供一套**从数据到模型再到应用**的 LLM 全栈实战指南。课程核心聚焦于**数据工程**，并将其深度融入到模型训练的每一个环节——从底座预训练 (Pre-training) 到指令微调 (SFT)，再到人类价值观对齐 (RLHF/GRPO)。

课程不仅涵盖了纯文本、多模态、Embedding 等多维度的技术原理，更引入了**系统工程**视角，详解分布式训练、高效部署与模型合规。特别值得一提的是，本课程贯穿了 **LazyLLM** 全流程实战与 **Agent（智能体）** 的双重应用：既教授如何构建具备 Agent 能力的模型，也演示如何利用 Agent 自动化流水线来清洗和合成高质量数据，助力企业构建闭环的"数据飞轮"。

<iframe src="assets/course_map.html" width="100%" style="border:none; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); min-height: 800px;" onload="this.style.height = this.contentWindow.document.documentElement.scrollHeight + 'px'" scrolling="no"></iframe>

## 课程大纲

### 第一部分：大模型基础：架构、训练与数据范式 (2课时)
**目标**：深入剖析Transformer架构核心，确立"模型-训练-数据"三位一体的工程认知。

-   **第1课时：Transformer 核心与 Self-Attention 深度剖析**
    -   Transformer 层级结构：Embedding（Tokenization、位置编码）→ Multi-Head Self-Attention（Q/K/V 计算、因果 Masking）→ MLP → 输出概率分布（temperature/top-k/top-p 采样）。
    -   **辅助架构特性**：残差连接、Layer Normalization、Dropout 对训练稳定性的作用。
    -   **位置编码详解**：绝对位置编码（正弦固定式与可学习嵌入），相对位置编码（RoPE 旋转矩阵与 ALiBi 线性偏置），外推性问题深度剖析与 Linear Scaling。
    -   **主流模型架构演进**：BERT/GPT/T5 三大范式对比与选型建议；从 GPT-3 规模化增长到 LLaMA/Mistral 工程精简，再到 MoE 稀疏激活（DeepSeek/Mixtral）及长上下文优化（Qwen3）。
-   **第2课时：LLM 训练范式与数据工程全景**
    -   **训练范式全流程**：Pre-training（自回归 NLL 目标）→ SFT（指令-响应条件建模）→ 对齐训练，三阶段数学目标函数详解。
    -   **对齐方法演进**：RLHF+PPO（奖励模型+强化学习）→ DPO（偏好对直接优化）→ GRPO（组级相对优势）的脉络与差异。
    -   **Chinchilla 法则**：参数规模 $N$、数据规模 $D$ 与算力预算 $C$ 的三角关系，$D \approx kN$ 的工程意义与对训练策略的影响。
    -   **Data-Centric AI**：从"以模型为中心"到"以数据为中心"的范式转变，数据质量对大模型能力上限的决定性作用。
    -   **数据工程全生命周期**：采集 → 清洗 → 过滤 → 去重 → 扩展/合成 → 质量评估 → 配比 → 课程学习（Curriculum Learning）。

### 第二部分：系统工程基础：分布式、部署与Agent工具 (4课时)
**目标**：在处理数据前，先理解大规模训练系统的底座，掌握模型部署核心技术，并将Agent作为数据处理的基础工具，最后通过LazyLLM实践全流程。

-   **第3课时：分布式训练技术概览**
    -   **显存墙与时间墙**：分布式训练的双重驱动力，以 GPT-3 为例量化分析显存需求（16~20 Bytes/参数）与单卡训练时间。
    -   **通信原语**：AllReduce（Ring 算法）、AllGather、ReduceScatter 三大原语的语义、实现机制与典型应用场景（DDP/FSDP 中的梯度同步）。
    -   **硬件拓扑**：NVLink/NVSwitch 机内互联（900 GB/s）与 InfiniBand/RoCE 机间互联（RDMA 超低延迟）对并行策略选型的影响。
    -   **并行策略**：DDP（模型复制+AllReduce 梯度）vs FSDP/ZeRO（参数+梯度+优化器状态全分片）；TP 张量并行（层内矩阵切分）；PP 流水线并行（层间切分+Micro Batch 填充+1F1B）；3D 并行（DP+TP+PP）；MoE 专家并行（All-to-All 路由）。
    -   **显存优化**：ZeRO-1/2/3 系列、FlashAttention-2/3（IO 感知分块计算）、Gradient Checkpointing（重计算换显存）、CPU Offload。
    -   **分布式对数据的影响**：GBS 线性缩放律、Micro Batch 流水线填充、分布式 Sharding 与断点续训（含 RNG 状态恢复）。
-   **第4课时：模型部署与推理加速**
    -   **量化原理与方法对比**：线性量化数学推导；AWQ（激活感知通道缩放，INT4）、GPTQ（分组贪心量化+Hessian 误差校正）、SmoothQuant（激活-权重难度迁移实现 INT8 全量化）三种方案的原理与适用场景。
    -   **vLLM 与 PagedAttention**：KV Cache 内存碎片问题根源，借鉴 OS 分页机制的 Block Pool 管理，实现零拷贝、高利用率的 KV 管理。
    -   **吞吐优化三大技术**：Continuous Batching（动态调度，吞吐提升 10-20×）；Speculative Decoding（草稿模型+目标模型联合验证，延迟降低 2-3×）；KV Cache 量化（INT8/FP8，显存节省~50%）。
    -   **实践**：使用 LazyLLM 的 `deploy.vllm` 方法部署 Qwen 模型，对比原始模型与 AWQ 量化版本的加载耗时与推理吞吐。
-   **第5课时：基于Agent的数据处理**
    -   **Agent 策略详解**：Function Call Agent（直接调用工具循环）、ReAct（思考-行动-观察三元循环）、PlanAndSolve（先分解任务再动态执行）、ReWOO（无观察全量规划后综合反馈）的工作流程与对比。
    -   **MCP 协议（Model Context Protocol）**：Anthropic 提出的开放标准，客户端-服务器架构解析，LazyLLM 的直接接入与一键部署 MCP Server 实践。
    -   **Multi-Agent 数据处理架构**：清洗 Agent（语义级去噪）、标注 Agent（结构化标注+CoT生成）、审查 Agent（质量守门人）的分工协作体系。
    -   **实践**：基于 LazyLLM 构建 PDF→QA 自动化流水线（PDF解析→Chunk合并→图片提取→QA生成→质量打分→过滤→Alpaca/多模态格式转换），配合 SFT 微调验证数据构建质量。
-   **第6课时：基于 LazyLLM 的数据-训练-推理全流程实践**
    -   **数据流**：从 WikiText 连续文本出发，通过 LazyLLM 的 `build_phi4_pt_pipeline` 自动合成 Phi-4 风格 QA 数据，将原始文本转化为"Question–Answer"结构以提升语义信息密度。
    -   **模型训练**：使用 LazyLLM 封装 LLaMA-Factory，配置全参数继续预训练（`stage='pt'`），Cosine 学习率调度，BF16 混合精度，监控 Loss 曲线。
    -   **评测体系**：构造 `prefix → continuation` 评测任务，以 PPL（困惑度）衡量语言建模能力，交叉熵 Loss 衡量拟合程度，字符级 2-gram F1 衡量生成与参考的局部重合。

### 第三部分：预训练数据工程与实战 (Pre-training) (4课时)
**目标**：遵循"原理-数据-实战"逻辑，掌握从底座模型训练原理到海量数据构建的全流程技术。

-   **第7课时：预训练原理、策略与评测**
    -   **预训练核心原理**：自监督学习范式，Next Token Prediction（自回归 NLL 目标函数详解），模型如何通过"猜词游戏"学习语法、语义与世界知识。
    -   **预训练任务类型**：MLM（BERT 式，15% 遮盖+替换策略）、CLM（GPT 式自回归，序列内多目标）、多任务混合训练（统一格式+混合损失 $\mathcal{L}_{total} = \sum \lambda_i \mathcal{L}_i$）。
    -   **MoE 路由机制**：Top-k 稀疏激活，负载均衡正则项（CV² 变异系数），路由噪声与梯度传播，以 Switch Transformer/DeepSeek 为例说明优势与挑战。
    -   **训练策略调优**：Cosine Decay+Warmup 与 WSD 学习率调度对比；AdamW（自适应梯度+权重衰减）与 Lion（符号梯度，省显存）优化器原理；范数裁剪防止梯度爆炸。
    -   **模型评测**：PPL 计算全流程（logits→softmax→条件概率→PPL）；MMLU（57学科选择题）、GSM8K（多步数学推理）、HumanEval（Pass@k 代码评测）基准详解；Zero/Few-shot 评测策略与结果解读规范。
-   **第8课时：预训练数据构建全流程**
    -   **语料来源分类**：通用文本（CommonCrawl/网页/书籍/百科/论坛）与专用文本（多语言、科学论文 arXiv、代码 GitHub）的特点与差异化贡献。
    -   **数据清洗全流程**：规则过滤（语言识别、统计特征、关键词/模板过滤）；分类器过滤（FastText/KenLM 困惑度打分、LLM 质量评分）；PII 移除（正则+NER）。
    -   **去重技术**：精确去重（MD5/SimHash）与近似去重（MinHash + LSH，Bloom Filter），去重对 Scaling Laws 的影响。
    -   **质量评估体系**：基于统计规则的启发式过滤 vs 基于模型打分的质量分类器，数据分级体系设计。
    -   **Tokenizer 原理**：BPE（字节对编码）与 Unigram（概率分割）对比；词表大小选择、词表扩充对多语言能力的影响；特殊 token 设计原则。
    -   **数据配比与退火策略**：多域数据混合比例设计，退火阶段（Annealing）高质量数据上采样策略。
-   **第9课时：基于 LazyLLM 的预训练全链路实战**
    -   **数据准备**：使用 LazyLLM 的 `build_text_pt_pipeline` 对 WikiText 语料进行规则清洗、长度过滤、去重，将原始文本整理为结构化 chunk。
    -   **训练集构造**：取前 3000 条 chunk 组织为纯文本样本，直接驱动继续预训练任务。
    -   **评测集构造**：取前 200 条 chunk 按 `PREFIX_RATIO=0.6` 分割为 `prefix → continuation` 样本对，用于量化评测。
    -   **训练启动**：LazyLLM 封装 LLaMA-Factory，配置分布式预训练参数（全参数更新、Cosine 调度、BF16），监控训练 Loss 曲线。
    -   **自动化评测**：分别计算 PPL（困惑度）、交叉熵 Loss 与 2-gram F1，对比预训练前后模型的语言建模能力变化。
-   **第10课时：多模态 LLM 架构与预训练实战**
    -   **多模态架构范式**：从纯文本 LLM 演进动机出发，解析 Vision Encoder（ViT，$N=HW/P^2$ patch 切分）+ 跨模态对齐模块（线性投影/MLP/Q-Former）+ LLM 的三段式结构，以 LLaVA、Qwen-VL 为代表实例。
    -   **图文数据集构建**：LAION/COYO 等图文对数据来源；alt-text 清洗与 caption 合成（Synthetic Captions）策略；分辨率处理（任意分辨率 patch 切分 vs 固定 Pad-Resize）。
    -   **多模态预训练策略**：Stage-1 视觉-语言对齐预训练（冻结 LLM，仅训练投影层）；Stage-2 指令微调（全参数解冻，引入任务指令数据）。
    -   **LazyLLM 实战**：基于 Flickr30k 数据集构建图文对训练样本，配置多模态预训练流水线（数据准备 → 模型组装 → 训练启动），评测图文理解能力。

### 第四部分：微调数据工程与实战 (SFT) (4课时)
**目标**：遵循"原理-数据-实战"逻辑，掌握指令微调的核心原理、通用数据构建及多模态微调实战。

-   **第11课时：指令微调原理与策略**
    -   **SFT 本质与意义**：预训练"续写模式"到"指令执行模式"的转变，对齐税（Alignment Tax），SFT 如何激活模型隐含的指令遵循能力。
    -   **训练目标与数据格式**：SFT 损失仅对 Response 部分计算，System/User/Assistant 角色划分与多轮对话历史的条件建模。
    -   **PEFT 参数高效微调**：LoRA（低秩矩阵分解，$W = W_0 + \alpha AB$，秩 $r$、缩放因子 $\alpha$）；QLoRA（4-bit NF4 量化基座+BF16 LoRA 适配器+双量化+分页优化器，突破显存限制）；AdaLoRA（SVD 自适应秩分配）；DoRA（方向-幅度分解，仅更新方向分量）。
    -   **全量微调**：适用场景与局限，结合 ZeRO-3 Offload 突破显存约束的工程策略。
    -   **防止过拟合**：NEFTune（训练时向 embedding 注入均匀噪声，提升泛化）；Pack 训练（多样本合并填充 context window，避免 padding 浪费，提升吞吐）。
-   **第12课时：通用指令数据构建、合成与蒸馏**
    -   **指令数据类型与规范**：指令-响应、QA、多轮对话三类样本的构建规范；Alpaca（instruction/input/output）、ShareGPT（conversations 字段）、CoT 思维链格式的 Schema 设计与对比。
    -   **经典数据集解析**：GSM8K（多步数学 CoT）、OpenHermes（多任务通用）、WizardLM（Evol-Instruct 生成）等典型指令数据集的来源、格式与质量分析。
    -   **数据蒸馏（Distillation）**：Self-Instruct 方法论，利用强模型（Teacher LLM）从种子指令出发自动生成大规模合成指令数据；基于 RAG 的文档驱动 QA 对生成。
    -   **Evol-Instruct（数据进化）**：In-breadth（多样性横向扩展）与 In-depth（复杂度逐步提升）两种进化策略，以 WizardLM 为范例。
    -   **SFT 数据质量评估**：IFD（Instruction Following Difficulty）指标计算方法与基于模型打分的难度加权采样策略。
    -   **数据合成实战**: 基于数据流水线的SFT问答对合成。
-   **第13课时：基于 LazyLLM 的微调全链路实战**
    -   **实验任务**：以"安全拒答"为主线，训练模型在面对暴力、网络攻击、隐私泄露等恶意指令时输出合规拒绝，同时保留正常问题的回答能力。
    -   **数据生成 Pipeline**：从 HuggingFace WildJailbreak 数据集流式下载 10000 条英文恶意提示，经 LazyLLM `Text2QA Pipeline`（TextToChunks → QA 生成 → QAScorer 评分 → 质量过滤 → Alpaca 格式转换）自动构建中文安全拒答训练数据。
    -   **SFT 微调流程**：LazyLLM 封装 LLaMA-Factory，配置 LoRA 参数进行监督微调，训练集/测试集划分与验证。
    -   **效果评测**：使用 LLM-as-Judge 对基座模型与 SFT 模型在测试集上的输出进行自动评分，量化拒答准确率提升。
-   **第14课时：多模态指令微调与实战**
    -   **视觉指令微调演化路径**：从 Image Caption（被动描述）到 VQA（图像+问题→精确答案）再到多模态 CoT（含中间推理步骤）和交错图文（Interleaved Image-Text）的渐进升级。
    -   **多模态数据格式规范**：chat 格式（messages 字段，role/content 结构，content 中嵌入 `<image>` 占位符），多轮视觉对话样本的角色规范化处理。
    -   **医疗 VQA 数据构建实战**：以医疗影像（CT/X-Ray/病理图）为例，使用 Qwen-VL 等大模型自动生成含临床推理过程的多模态 CoT 训练数据，经质量打分过滤后用于小模型微调。
    -   **LazyLLM 实战**：配置多模态 LoRA 微调流程，对比 SFT 前后模型在医学影像 VQA 测试集上的评分（LLM-as-Judge 自动打分）。

### 第五部分：对齐数据工程与实战 (Alignment) (4课时)
**目标**：遵循"原理-数据-实战"逻辑，掌握RLHF/GRPO核心算法、偏好数据构建、安全合规及对齐实战。

-   **第15课时：对齐算法原理 (RLHF & GRPO)**
    -   **强化学习基础**：智能体-环境-动作-状态-奖励框架在 LLM 对齐中的映射，SFT 的局限性（"模仿陷阱"）与 RL 的互补价值。
    -   **RLHF + PPO**：奖励模型（RM）训练流程，PPO 裁剪目标函数（防策略更新过猛），KL 散度约束（防偏离参考模型），以 InstructGPT 为标志性案例。
    -   **DPO（Direct Preference Optimization）**：去掉显式 RM 和 RL，直接在偏好对 $(y^+, y^-)$ 上优化概率比，$\beta$ 控制偏好强度；对比 IPO（无 Bradley-Terry 假设）、KTO（单样本好/坏标签）。
    -   **GRPO（Group Relative Policy Optimization）**：DeepSeek-R1 的核心算法，同一 prompt 下采样一组候选回答，以组内相对优势替代 Critic 网络估值，显著强化模型推理链能力。
-   **第16课时：偏好数据构建**
    -   **SFT 的根本局限**：最大似然估计只学"概率"不懂"价值"，幻觉温床与对开放式问题无法量化优劣，由 InstructGPT 1.3B 胜过 175B GPT-3 引出对齐的必要性。
    -   **偏好数据核心理论**：Bradley-Terry 模型（$P(y_w > y_l | x) = \sigma(r(x,y_w) - r(x,y_l))$），从"文本补全"转向"偏序比较"的范式革命。
    -   **偏好数据构建方法**：Pairwise（chosen/rejected 三元组）vs Listwise（排序列表）格式；人工标注 vs LLM-as-a-Judge（蒸馏评分流水线，标注一致性校验）。
    -   **过程奖励数据（PRM）**：Math-Shepherd 风格的 Step-by-step 验证，MC 采样估算中间步骤正确性，构建过程奖励训练样本。
    -   **规则奖励数据集**：基于答案正确性（GSM8K 精确匹配）与格式合规性（JSON 结构、长度约束）的 Rule-based Reward 自动构建方法。
-   **第17课时：模型风险、合规与伦理**
    -   **幻觉（Hallucination）**：NTP 概率优先于事实、训练数据污染、有损压缩三大技术根因；以加拿大航空 AI 客服法律赔偿案为真实案例，解析缓解方案（RAG、FactChecking、RLHF）。
    -   **偏见（Bias）与毒性（Toxicity）**：训练数据中的社会偏见如何被模型放大；毒性内容检测分类器与 RLHF 对齐策略。
    -   **越狱攻击与防御**：提示词攻击类型（角色扮演越狱/Base64 绕过/DAN/梯度攻击）；防御策略（输入过滤、对抗训练、Constitutional AI）。
    -   **红队测试（Red Teaming）**：自动化攻击提示词生成流程（AdvBench/HarmBench 基准），对抗样本构建与人机协同攻防实验设计。
    -   **合规与伦理**：GDPR/CCPA 数据隐私合规要求，训练数据版权审查（Books3 诉讼案例），国内 AI 监管法规，负责任 AI（公平/可解释/安全/隐私）四大原则。
-   **第18课时：基于 LazyLLM 的对齐全链路实战**
    -   **奖励模型（RM）训练**：使用 `trl` 库的 `RewardTrainer` 实现 Pairwise Ranking Loss，将分类模型转化为标量打分器，基于 Anthropic HH-RLHF 偏好数据（prompt/chosen/rejected）训练。
    -   **PPO 强化学习训练**：以 CartPole 环境直观演示 RL 基本循环，延伸到 LLM 的四模型架构（策略/参考/奖励/价值模型），配置 PPOConfig 完成 RLHF 对齐。
    -   **DPO 直接偏好优化**：DPO 完整训练流程，在偏好对上直接优化概率比，无需奖励模型，工程更简洁。
    -   **GRPO 组级优化实战**：以 GSM8K 答案正确性作为规则奖励信号，训练模型数学推理能力。
    -   **效果对比评测**：对比对齐前后模型在安全拒答、指令遵循、数学推理三个维度上的表现差异。


### 第六部分：特定领域能力数据构建 (Domain-Specific Data) (6课时)
**目标**：深入垂直领域，掌握代码、数学、Agent等高难度数据的构建，以及行业模型的全流程训练。

-   **第19课时：推理与数学能力增强**
    -   **推理数据（CoT）构建**：Zero-shot CoT、Few-shot CoT、Auto-CoT 与 Self-Consistency（多路采样投票）的方法论；Magpie 风格从模型自生成 CoT 数据的大规模合成策略。
    -   **多跳推理数据构建**：多文档实体链构建，跨段落依赖关系设计，复杂逻辑推理路径的合成方法。
    -   **数学数据集构建**：Step-by-step 格式标准化、LaTeX 公式规范化；利用 SymPy/Z3 符号求解器自动验证数学步骤正确性；正确路径 vs 错误路径负样本对构建（过程监督数据）。
    -   **数据合成与增强**：Evol-Instruct 在数学领域的应用（增加推理步骤/引入干扰条件/问题类型转换），合成新数学题，去重与质量过滤流程。
    -   **LazyLLM 实战**: 通过流水线数据生成与微调强化模型数学与推理能力。
-   **第20课时：代码能力增强**
    -   **代码数据集构建**：GitHub 仓库筛选策略（Stars、License、活跃度），文件级过滤（排除自动生成代码、敏感信息），MinHash+LSH 近似去重。
    -   **依赖解析与拓扑排序**：代码文件间依赖图构建（import/require 解析），DFS 拓扑排序重组文件顺序，保持函数调用链与上下文的逻辑连贯性。
    -   **Fill-in-the-Middle（FIM）预训练任务**：PSM/SPM 格式设计（Prefix-Suffix-Middle token 重排），FIM 任务对代码补全能力的核心价值与训练配比。
    -   **代码指令微调数据**：代码生成（问题描述→代码实现），单元测试生成（代码→测试用例），代码注释与文档生成的数据构造方法。
    -   **执行反馈（Code RL）**：Docker 沙箱隔离执行环境，以编译通过率/测试通过率（Pass Rate）作为 Reward，构建自动化强化学习循环，实现验证驱动的数据过滤与模型迭代。
-   **第21课时：长上下文能力增强**
    -   **长上下文核心挑战**：从 token 级长度与文档级语义复杂度两个维度解析问题，剖析位置编码外推失效、注意力稀释与关键信息遗忘三大瓶颈。
    -   **长文本数据构建**：书籍（Project Gutenberg/BooksCorpus）、学术论文（arXiv）、财务报告、法律合同等天然长文档的选取策略，跨文档拼接（保留实体/主题关联），段落级滑动窗口，章节级分组。
    -   **"大海捞针"（NIAH）合成数据**：在长文档随机位置插入"针"（关键事实），构造 `(长文档上下文, 针位置问题, 答案)` 三元组训练样本，提升模型对长窗口任意位置的精确检索能力。
    -   **建模与评测方法**：长窗口建模（如 RoPE 扩展与训练策略）、长序列计算与 KV Cache 开销、显存友好注意力机制（分块、Flash、Ring Attention），以及长上下文训练技巧、评测体系与能力边界。
    -   **LazyLLM 评测实战**：基于 SQuAD 2.0 构建长上下文评测集，对比基座模型与微调模型表现；通过多参考答案指标（ROUGE-L、Token-F1、EM）进行效果评估，并结合 PPL 随序列长度变化的趋势分析模型在长上下文下的性能退化。
-   **第22课时：结构化输出与格式对齐**
    -   **必要性与挑战**：概率生成 vs 确定性系统的核心矛盾；格式错误、内容幻觉、类型不匹配三大痛点。
    -   **结构化抽取数据集生态**：三段式样本结构（指令+Schema+文本）；开源数据集全景：通用抽取、垂直领域、代码查询、指令微调等。
    -   **数据构建流水线**：Schema 设计（JSON Schema/Pydantic 字段定义）→ 逆向合成（先生成目标 JSON 再反推自然语言描述）→ 自动化校验（jsonschema 验证）→ 负样本构建（字段缺失/类型错误/格式违规样本）。
    -   **训练与推理技术**：提示工程（Few-shot、TypeScript风格Schema）；监督微调（边界标记、格式错误清除）；推理时约束（Trie树Token剪枝、语法状态机掩码）。
    -   **评测指标与方法**：格式错误率、字段级准确率、幻觉率；自动化评测四阶段（解析→对齐→判定→汇总）。
    -   **LazyLLM 实战**：Text2SQL 流水线与中文医疗实体关系抽取 CMEIE 完整实验。
-   **第23课时：Agent 能力增强 (Tools & Planning)**
    -   **Agent 范式转移**：从被动 Chatbot 到主动 Agent（感知-规划-执行-反馈闭环），"思维-行动"轨迹数据稀缺性的根本原因（隐性推理过程缺失、环境不可复制）。
    -   **工具调用数据集构建**：ToolBench、WebArena、SWEbench 等经典数据集解析；API 定义（OpenAPI 格式）、参数生成、多步调用轨迹（ReAct/FunctionCall 格式）的数据结构设计。
    -   **规划能力数据构建**：合成 ReAct（思考-行动-观察三元组序列）与 Plan-and-Solve（计划分解→子任务执行→结果综合）多步规划轨迹；SimIA 框架利用模拟环境自动标注交互数据。
    -   **RAT（Retrieval-Augmented Thinking）**：在工具调用链中引入检索增强推理，提升工具参数生成的准确性与可靠性。
    -   **多轮状态保持**：对话历史中工具调用状态的序列化，环境反馈模拟（tool output 注入与错误恢复训练数据设计）。
-   **第24课时：行业领域模型实战**
    -   **通用模型局限与行业需求**：知识陈旧、术语错误、缺乏专业输出结构、幻觉风险，以医疗/法律/金融为例说明"百科大学生"到"持证专家"的差距。
    -   **行业数据集准备**：垂直领域数据清洗（去噪、格式归一）；医疗病历与金融报表的 PII 脱敏策略；知识图谱融合（将 KG 三元组转化为文本训练对，注入结构化领域知识）。
    -   **继续预训练（CPT）策略**：学习率选择（比初始预训练低约 10 倍），通用语料与领域语料配比（通用:领域 = 3:7～5:5），混入通用数据缓解灾难性遗忘。
    -   **领域指令数据集构建**：医疗（病历问答/症状诊断/药物交互）、法律（合同审查/法规检索）、金融（财报分析/风险评估）场景的指令数据设计，基于领域文档自动生成 QA 对。
    -   **LazyLLM 实战**：CPT → SFT 两阶段训练流水线，使用领域专属 PPL 和下游任务准确率量化领域注入效果。


### 第七部分：检索增强生成 (RAG) 数据工程 (4课时)
**目标**：理解 RAG 核心架构，掌握从文档处理到 Embedding/Reranker 模型微调的全流程数据工程。

-   **第25课时：RAG 架构原理与数据处理**
    -   **RAG 核心范式**：解决 LLM 幻觉与知识固化的工业级方案，RAG vs Long Context 的优劣对比（知识更新成本/精度/适用场景），"开卷答题"隐喻。
    -   **RAG 三阶段全流程**：检索（BM25 稀疏检索+向量密集检索+混合检索）→ 增强（多源文档归一、语义分块策略、Metadata 增强）→ 生成（Prompt 模板组装+LLM 生成+答案后处理）。
    -   **文档智能处理**：PDF/Word/HTML 多源格式转换，固定大小/语义/递归分块策略对比，标题-来源-日期 Metadata 注入，LLM 驱动的 Agent 自动摘要与元数据增强入库。
    -   **多跳 QA 数据生成**：Atomic（单跳原子问答）、Depth（深度多跳推理）、Width（宽度多证据融合）三类合成模式，LLM 蒸馏精炼与质量过滤。
    -   **LazyLLM 实战**：完整知识库构建与 RAG 问答系统搭建，验证 Embedding/Reranker 微调的必要性。
-   **第26课时：Embedding 模型微调与实战**
    -   **通用 Embedding 的语义错位**：医疗/法律/金融三类典型误召回案例，根因在于通用模型未学习领域特有语义边界。
    -   **Bi-encoder 架构与对比学习**：双塔独立编码（推理高效）vs Cross-encoder 联合编码（精度高但慢），InfoNCE 损失（$\mathcal{L} = -\log \frac{e^{q \cdot d^+/\tau}}{\sum e^{q \cdot d_i/\tau}}$），温度参数 $\tau$ 对分布锐度的影响。
    -   **难负样本（Hard Negatives）挖掘**：随机负样本的局限性，BM25 召回后用 Cross-encoder 重排挑选难负样本，AugSBERT/LLM 生成难负样本对的工程方法。
    -   **LazyLLM 实战**：微调 BGE/E5 等 Embedding 模型，在 MTEB（检索/聚类/分类子任务）上量化领域提升效果，并集成到 LazyLLM RAG 知识库进行端到端检索对比。
-   **第27课时：Reranker 模型微调与实战**
    -   **Reranker 精排原理**：初筛（Bi-encoder 召回）→ 精排（Reranker 重排）两阶段 RAG 架构，Cross-encoder 通过 Query-Document 拼接联合 Transformer 编码实现深度语义交互，对比 Bi-encoder 的精度优势。
    -   **Cross-encoder 打分机制**：`[CLS] query [SEP] document [SEP]` 输入格式，`[CLS]` 向量→分类头→相关性概率，以 MacBERT/BERT 为骨架的实现细节。
    -   **排序训练数据构建**：Pairwise 格式（query/positive/negative 三元组，Margin Ranking Loss）与 Listwise 格式（排序列表，Listwise Softmax Loss）对比；利用 LLM 对候选文档打分生成伪标签，蒸馏高质量排序数据。
    -   **LazyLLM 实战**：微调 BGE-Reranker，搭建"Bi-encoder 初筛 + Reranker 精排"完整 RAG 流水线，以召回率/MRR/NDCG 量化检索效果提升。
-   **第28课时：Agentic RAG 与多跳数据增强**
    -   **RAG 代际演进**：朴素 RAG → 高级 RAG（查询改写+重排序）→ 模块化 RAG → Agentic RAG（LLM 作为规划控制器，主动决定何时检索、用何工具、如何根据中间结果调整策略）。
    -   **多跳 QA 挑战解析**：推理链长度 L>1 时单次检索的失效原因，噪声段落干扰与跨文档证据整合难题，HotpotQA 等基准的支持句标注机制。
    -   **三类数据合成策略**：Atomic（原子单跳，基础检索-回答对）；Depth（深度多跳，bridge question 推理链，逐跳答案验证）；Width（宽度多证据，parallel question，多文档融合回答）的构建流程与质量控制。
    -   **Grounding 约束与 Self-RAG**：有依据标签（答案必须来源于检索文档），检索意图识别（何时需要检索）、文档相关性反思（IsRel）、回复生成质量打分（IsSup）数据的构建思路。
    -   **LazyLLM 实战**：使用 `atomic_rag_pipeline` 生成多跳增强 JSONL，`TrainableModule + finetune.auto` 指令微调，在同一验证集上以 F1/EM 对比微调前后的多跳问答能力。
