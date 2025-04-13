from lazyllm.tools.rag import Document

doc = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb")

data = doc._impl._reader.load_data(input_files=["平安证券-珀莱雅.pdf"])
print(f"data: {data}")
print(f"text: {data[0].text}")
print(f"text: {data[1].text}")
