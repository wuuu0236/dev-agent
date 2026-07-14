"""
Embedding 模块：通过 API 调用，不下载本地模型

之前的问题是 Streamlit Cloud 每次冷启动要重新下载 470MB 的
text2vec-base-chinese 模型，导致文档上传时进度条卡死。

现在改用硅基流动 Embedding API（OpenAI 兼容格式），发 HTTP 请求
就能拿到向量，零下载，秒级响应。
"""
from openai import OpenAI
from src.config import EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL

_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量将文本转成向量（走 API）"""
    if not texts:
        return []
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [d.embedding for d in response.data]


def embed_single(text: str) -> list[float]:
    """单个文本转向量"""
    return embed_texts([text])[0]
