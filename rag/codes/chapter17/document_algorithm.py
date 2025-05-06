import time
from lazyllm.tools import Document
from lazyllm import OnlineEmbeddingModule, Retriever
import requests
import io

path = "path/to/docs"
docs = Document(path, manager='ui', embed=OnlineEmbeddingModule())
# 注册分组
Document(path, name='法务文档管理组', manager=docs.manager)
Document(path, name='产品文档管理组', manager=docs.manager)
#  模拟文档上传
docs.start()
manager_url = docs._manager.url.rsplit('/', 1)[0]
def get_url(manager_url, **kw):
    url = f"{manager_url}/add_files_to_group"
    if kw: url += ('?' + '&'.join([f'{k}={v}' for k, v in kw.items()]))
    return url

files = [('files', ('产品文档.txt', io.BytesIO("这是一篇产品文档。这是产品部的文档。\n来自产品文档管理组".encode("utf-8")), 'text/plain'))]
data = dict(override='true', group_name="产品文档管理组")
response = requests.post(get_url(manager_url, **data), files=files)
assert response.status_code == 200
time.sleep(5)

files = [('files', ('法务文档.txt', io.BytesIO("这是一篇法务文档。这是法务部的文档。\n来自法务文档管理组".encode("utf-8")), 'text/plain'))]
data = dict(override='true', group_name="法务文档管理组")
response = requests.post(get_url(manager_url, **data), files=files)
assert response.status_code == 200
time.sleep(5)

# 为不同的文档组设置不同算法并进行检索 
doc1 = Document(path, name='法务文档管理组', manager=docs.manager)
doc2 = Document(path, name='产品文档管理组', manager=docs.manager)

# 法务文档管理组 按照 \n 进行切分，产品文档管理组按照 。 进行切分
doc1.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
doc2.create_node_group(name="line", transform=lambda s: s.split("。") if s else '')


retriever1 = Retriever([doc1], group_name="block", similarity="cosine", topk=3)
retriever2 = Retriever([doc2], group_name="line", similarity="cosine", topk=3)

for node in retriever1("法务"):
    print(node.text)
    print('==========')


for node in retriever2("产品"):
    print(node.text)
    print('==========')

