import lazyllm
from lazyllm import OnlineChatModule, pipeline
from lazyllm.tools import IntentClassifier

from statistical_agent import build_statistical_agent
from paper_rag_multimodal import build_paper_rag
from mcp_agent import paper_agent, calculator_agent

gen_prompt = """
# 图片信息提取器

1. 返回格式：
   ### 提问: [用户原问题]
   ### 提问中涉及到的图像内容描述：[客观描述，包括主体、背景、风格等]
2. 要求：详细、中立，避免主观猜测

**示例：**
输入："找类似的猫图"（附橘猫图）,
响应如下：
   ### 提问: 找类似的猫图
   ### 提问中涉及到的图像内容描述：一只橘猫趴在沙发上，阳光从左侧照射，背景是米色窗帘

"""

# 构建 rag 工作流和统计分析工作流
rag_ppl = build_paper_rag()
sql_ppl = build_statistical_agent()

# 搭建具备知识问答和统计问答能力的主工作流
def build_paper_assistant():
    llm = OnlineChatModule(source='qwen', stream=False)
    vqa = lazyllm.OnlineChatModule(
        source="sensenova",
        model="SenseNova-V6-Turbo").prompt(lazyllm.ChatPrompter(gen_prompt))

    with pipeline() as ppl:
        ppl.ifvqa = lazyllm.ifs(
            lambda x: x.startswith('<lazyllm-query>'),
            lambda x: vqa(x), lambda x: x)
        with IntentClassifier(llm) as ppl.ic:
            ppl.ic.case["论文问答", rag_ppl]
            ppl.ic.case["统计问答", sql_ppl]
            ppl.ic.case["计算器", calculator_agent]
            ppl.ic.case["网页最新论文搜索", paper_agent]

    return ppl

if __name__ == "__main__":
    main_ppl = build_paper_assistant()
    lazyllm.WebModule(main_ppl, port=23459, static_paths="./images", encode_files=True).start().wait()
