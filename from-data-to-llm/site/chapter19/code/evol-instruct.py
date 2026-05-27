from lazyllm import OnlineChatModule

base_template = '''
你是一个问题进化（evol-instruct）Agent，
需要在保持原问题核心语义不变的前提下，对问题进行复杂化改写。

改写规则：
{constraints}

原始问题：
{question}

请输出改写后的问题，不要解释。
'''

constraints = []


def add_constraint(c):
    constraints.append(c)


add_constraint('保持原问题核心语义不变')
add_constraint('增加问题的语义复杂度')
add_constraint('引入背景或隐含前提')
add_constraint('增加逻辑关系或推理要求')
add_constraint('使问题无法用一句话回答')

model = OnlineChatModule()

query = base_template.format(
    constraints='\n'.join([f'- {c}' for c in constraints]),
    question='何为道？'
)

print(model(query))
