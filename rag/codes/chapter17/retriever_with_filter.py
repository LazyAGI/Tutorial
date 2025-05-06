import time
import uuid
import os
import requests
import io
import json

from lazyllm import OnlineEmbeddingModule
from lazyllm.launcher import cleanup
from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc as DocField
from lazyllm.tools.rag import DataType

def get_milvus_store_conf(rag_dir: str = os.path.abspath('./data/milvus'), kb_group_name: str = str(uuid.uuid4())):
    milvus_db_dir = os.path.join(rag_dir, kb_group_name)
    if not os.path.exists(milvus_db_dir):
        os.makedirs(milvus_db_dir)

    milvus_store_conf = {
        'type': 'milvus',
        'kwargs': {
            'uri': os.path.join(milvus_db_dir, "milvus.db"),
            'index_kwargs': [
                {
                    'embed_key': 'dense',
                    'index_type': 'IVF_FLAT',
                    'metric_type': 'COSINE',
                },
            ]
        },
    }
    return milvus_store_conf


def upload_files(doc):
    doc.start()
    manager_url = doc._manager.url.rsplit('/', 1)[0]
    def get_url(manager_url, **kw):
        url = f"{manager_url}/upload_files"
        if kw: url += ('?' + '&'.join([f'{k}={v}' for k, v in kw.items()]))
        return url

    files = [('files', ('合同问题1.txt', io.BytesIO("1.合同问题，涉及条款修订、履约风险及争议解决，需尽快审核。".encode("utf-8")), 'text/plain')),
             ('files', ('合同问题2.txt', io.BytesIO("2.合同问题，包括付款延迟、违约责任和保密协议，需总监审批。".encode("utf-8")), 'text/plain'))]
    data = dict(override='true', metadatas=json.dumps([{"department": "法务一部"},
                                                       {"department": "法务二部"}]))
    response = requests.post(get_url(manager_url, **data), files=files)
    assert response.status_code == 200
    time.sleep(20)


if __name__ == "__main__":
    Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)
    CUSTOM_DOC_FIELDS = {"department": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
    doc = Document(os.path.abspath("./data/rag_data_filter"), name='law_kg', doc_fields=CUSTOM_DOC_FIELDS,
                    embed={"dense": OnlineEmbeddingModule(source="qwen")}, manager=True,
                    store_conf=get_milvus_store_conf(kb_group_name='law_kg'))
    retriever = Retriever(doc, group_name="sentences", topk=5, embed_keys=['dense'])
    # 上传文件
    upload_files(doc)


    while True:
        query = input("input your query:")  # 合同问题
        department = input("input you department:")
        params = {"query": query}
        if department:
            params |= {"filters": {"department":[department]}}
        nodes = retriever(**params)
        print("=========== QUERY =============")
        print(query)
        print("============ RESULTS ==========")
        for node in nodes:
            print(node.text)
            print()
        print("==============================\n")
