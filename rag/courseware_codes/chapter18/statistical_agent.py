import lazyllm
from lazyllm import pipeline
from lazyllm.tools.agent import ReactAgent

from utils import bi_tools


def build_statistical_agent():
    with pipeline() as sql_ppl:
        sql_ppl.formarter = lambda query: bi_tools.BI_PROMPT.format(query=query, image_path="./images")
        sql_ppl.agent = ReactAgent(
            llm=lazyllm.OnlineChatModule(source='qwen', model='qwen-max-latest', stream=False),
            tools=['run_code', 'run_sql_query'],
            return_trace=True,
            max_retries=3)
        sql_ppl.clean = lazyllm.ifs(lambda x: "Answer:" in x, lambda x: x.split("Answer:")[-1], lambda x: x)

    return sql_ppl


if __name__ == "__main__":
    main_ppl = build_statistical_agent()
    lazyllm.WebModule(main_ppl, port=20012, static_paths="./images").start().wait()
