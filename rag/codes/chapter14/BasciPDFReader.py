import os
import lazyllm

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(lazyllm.config['home'], 'rag_for_qa'))
        self.rag_dir = os.path.join(self.root_dir, "rag_master")
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")
        self.image_path = "/home/workspace/LazyRAG-Enterprise/images"

tmp_dir = TmpDir()
documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir)
documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

retriever = lazyllm.Retriever(documents, group_name="block", similarity="bm25", topk=3, output_format='content')

print(retriever('deepseek-r1相关论文的Abstract'))
