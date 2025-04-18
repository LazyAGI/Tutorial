import os
import lazyllm

prompt = ('You will act as an AI question-answering assistant and complete a dialogue task.'
          'In this task, you need to provide your answers based on the given context and questions.')

embed = lazyllm.TrainableModule(
    'bge-large-zh-v1.5',
    os.path.join(os.getcwd(), 'save_ckpt/path/to/sft/embed/model'))

documents = lazyllm.Document(dataset_path=os.path.join(os.getcwd(), "KB"), embed=embed, manager=False)
documents.create_node_group(name='split_sent', transform=lambda s: s.split('\n'))
with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(
        doc=documents, group_name="split_sent", similarity="cosine", topk=1, output_format='content', join='')
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | lazyllm.bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(source="sensenova")\
        .prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

ppl.start()

while True:
    query = input('\n\n===========\n请输入您的问题:\n')
    res = ppl(query)
    print(f'\n=== RAG Answer:\n{res}')
