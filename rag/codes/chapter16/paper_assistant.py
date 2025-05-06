import lazyllm
from lazyllm import OnlineChatModule, pipeline, _0
from lazyllm.tools import IntentClassifier

from statistical_agent import build_statistical_agent
from paper_rag import build_paper_rag

# 构建 rag 工作流和统计分析工作流
rag_ppl = build_paper_rag()
sql_ppl = build_statistical_agent()

# 搭建具备知识问答和统计问答能力的主工作流
def build_paper_assistant():
    llm = OnlineChatModule(source='qwen', stream=False)
    intent_list = [
        "论文问答",
        "统计问答",
    ]

    with pipeline() as ppl:
        ppl.classifier = IntentClassifier(llm, intent_list=intent_list)
        with lazyllm.switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
            ppl.sw.case[intent_list[0], rag_ppl]
            ppl.sw.case[intent_list[1], sql_ppl]

    return ppl

if __name__ == "__main__":
    main_ppl = build_paper_assistant()
    lazyllm.WebModule(main_ppl, port=23459, static_paths="./images").start().wait()
