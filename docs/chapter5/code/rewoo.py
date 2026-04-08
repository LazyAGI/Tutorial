import lazyllm
from lazyllm import fc_register, ReWOOAgent
import wikipedia


@fc_register('tool')
def WikipediaWorker(input: str):
    '''
    Find information in Wikipedia

    Args:
        input(str): search content
    '''
    print('Pedia Worker Called')
    try:
        evidence = wikipedia.page(input).content
        evidence = evidence.split('\n\n')[0]
    except wikipedia.PageError:
        evidence = (
            f'Could not find [{input}]. Similar: {wikipedia.search(input)}'
        )
    except wikipedia.DisambiguationError:
        evidence = (
            f'Could not find [{input}]. Similar: {wikipedia.search(input)}'
        )
    print(evidence)
    return evidence


@fc_register('tool')
def LLMWorker(input: str):
    '''
    Docstring for LLMWorker

    Args:
        input(str): Worker's input.
    '''
    print('LLMWorker Called')
    llm = lazyllm.OnlineChatModule(stream=False)
    query = f'Respond in short directly with no extra words.\n\n{input}'
    response = llm(query, llm_chat_history=[])
    return response


tools = ['WikipediaWorker', 'LLMWorker']
llm = lazyllm.OnlineChatModule()
agent = ReWOOAgent(llm, tools=tools)
query = (
    'What is the name of the cognac house that makes the main ingredient '
    f'in The Hennchata? You should call the tools to find the answer: {tools}'
)
ret = agent(query)
print(ret)
