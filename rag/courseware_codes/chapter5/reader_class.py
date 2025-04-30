from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag.readers.readerBase import infer_torch_device
from lazyllm.tools.rag import DocNode
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
from lazyllm.tools.rag import Document


class ImageDescriptionReader(ReaderBase):
    def __init__(self, parser_config: Optional[Dict] = None, prompt: Optional[str] = None) -> None:
        super().__init__()
        if parser_config is None:

            device = infer_torch_device()
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                                 torch_dtype=dtype)
            parser_config = {"processor": processor, "model": model, "device": device, "dtype": dtype}
        self._parser_config = parser_config
        self._prompt = prompt

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        model = self._parser_config['model']
        processor = self._parser_config["processor"]

        device = self._parser_config["device"]
        dtype = self._parser_config["dtype"]
        model.to(device)

        inputs = processor(image, self._prompt, return_tensors="pt").to(device, dtype)

        out = model.generate(**inputs)
        text_str = processor.decode(out[0], skip_special_tokens=True)
        return [DocNode(text=text_str, metadata=extra_info or {})]


doc = Document(dataset_path="your_doc_path")
doc.add_reader("*.png", ImageDescriptionReader)
data = doc._impl._reader.load_data(input_files=["cmrc2018_path/LazyLLM-logo.png"])
print(f"data: {data}")
print(f"text: {data[0].text}")
