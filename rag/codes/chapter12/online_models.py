from typing import Dict, List, Union

from lazyllm import OnlineChatModule, OnlineEmbeddingModule


from lazyllm.module import OnlineEmbeddingModuleBase

class CustomOnlineEmbeddingModule(OnlineEmbeddingModuleBase):
    """CustomOnlineEmbeddingModule"""

    def __init__(self, embed_url, embed_model_name, api_key, model_series):
        super().__init__(
            embed_url=embed_url, embed_model_name=embed_model_name,
            api_key=api_key, model_series=model_series
        )

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {"inputs": text, "model": self._embed_model_name}
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(
            self,
            response: Union[List[List[str]], Dict]
        ) -> Union[List[List[str]], Dict]:
        return response


DOUBAO_API_KEY = ""
DEEPSEEK_API_KEY = ""
QWEN_API_KEY = ""

llm = OnlineChatModule(
    source="deepseek",
    api_key=DEEPSEEK_API_KEY,
)

embedding_model = OnlineEmbeddingModule(
    source="doubao",
    embed_model_name="doubao-embedding-large-text-240915",
    api_key=DOUBAO_API_KEY,
)

custom_embedding_model = CustomOnlineEmbeddingModule(
    embed_url="",
    embed_model_name="BAAI/bge-m3",
    api_key="",
    model_series="bge"
)

rerank_model = OnlineEmbeddingModule(
    source="qwen",
    api_key=QWEN_API_KEY,
    type="rerank"
)
