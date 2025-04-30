import os
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import DocNode, Document

class YmlReader(ReaderBase):
    def _load_data(self, file, extra_info=None, fs=None):
        with open(file, 'r') as f:
            data = f.read()
            node = DocNode(text=data, metadata=extra_info or {})
            node._content = "Call the class YmlReader."
            return [node]

def processYml(file, extra_info=None):
    with open(file, 'r') as f:
        data = f.read()
        node = DocNode(text=data, metadata=extra_info or {})
        node._content = "Call the function processYml."
        return [node]

class TestRagReader(object):
    def __init__(self):
        self.datasets = os.path.join(os.getcwd(), 'test_reader')
        self.doc1 = Document(dataset_path=self.datasets, manager=False)
        self.doc2 = Document(dataset_path=self.datasets, manager=False)
        

    def test_register_local_reader(self):
        self.doc1.add_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        docs2 = self.doc2._impl._reader.load_data(input_files=files)
        print(f"docs1 {'calls' if docs1[0].text == 'Call the function processYml.' else 'does not call'} the function processYml")
        print(f"docs2 {'calls' if docs2[0].text == 'Call the function processYml.' else 'does not call'} the function processYml")


    def test_register_global_reader(self):
        Document.register_global_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        docs2 = self.doc2._impl._reader.load_data(input_files=files)
        print(f"docs1 {'calls' if docs1[0].text == 'Call the function processYml.' else 'does not call'} the function processYml")
        print(f"docs2 {'calls' if docs2[0].text == 'Call the function processYml.' else 'does not call'} the function processYml")

    def test_register_local_and_global_reader(self):
        files = [os.path.join(self.datasets, "reader_test.yml")]

        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        if docs1[0].text != "Call the class YmlReader." and docs1[0].text != "Call the function processYml.":
            print('docs1 does not call class YmlReader and function processYml') 
        Document.add_reader("**/*.yml", processYml)
        self.doc1.add_reader("**/*.yml", YmlReader)
        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        docs2 = self.doc2._impl._reader.load_data(input_files=files)
        if docs1[0].text == "Call the class YmlReader." and docs2[0].text == "Call the function processYml.":
            print('docs1 call class YmlReader and docs2 call function processYml')

test_reader = TestRagReader()
# test_reader.test_register_local_reader()
# test_reader.test_register_global_reader()
test_reader.test_register_local_and_global_reader()
