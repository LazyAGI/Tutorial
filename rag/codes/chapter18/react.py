import json
import lazyllm
from lazyllm import fc_register, ReactAgent

@fc_register("tool")
def search_knowledge_base(query: str):
    '''
    Get info from knowledge base in a given query.

    Args:
        query (str): The query for search knowledge base.
    '''
    return "无形"

llm = lazyllm.OnlineChatModule(stream=False)

tools = ["search_knowledge_base"]
agent = ReactAgent(llm, tools)

if __name__ == "__main__":
    res = agent("何为天道？")
    print("Result: \n", res)
