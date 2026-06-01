# from-data-to-llm 环境配置

本文面向协作者和自动化 agent，用于快速准备 `from-data-to-llm` 教程运行环境。

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
