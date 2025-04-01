from lazyllm.tools.rag import Document

doc = Document(dataset_path="your_doc_path")

data = doc._impl._reader.load_data(input_files=["webPage.html"])
print(f"data: {data}")
print(f"text: {data[0].text}")
