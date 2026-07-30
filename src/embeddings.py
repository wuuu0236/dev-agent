"""
Embedding 模块：通过 API 调用，不下载本地模型

之前的问题是 Streamlit Cloud 每次冷启动要重新下载 470MB 的
text2vec-base-chinese 模型，导致文档上传时进度条卡死。

现在改用硅基流动 Embedding API（OpenAI 兼容格式），发 HTTP 请求
就能拿到向量，零下载，秒级响应。
"""
import sys
from openai import OpenAI
from src.config import (
    EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL,
    EMBEDDING_BACKEND, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
)

_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE, timeout=30.0)
# 本地 Ollama embedding 客户端（仅 EMBEDDING_BACKEND=ollama 时使用）
_ollama_client = None


def _get_ollama_embed_client() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=60.0)
    return _ollama_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量将文本转成向量。

    EMBEDDING_BACKEND=cloud（默认）→ 硅基流动 API，零下载、秒级响应。
    EMBEDDING_BACKEND=ollama        → 本地 Ollama embedding（nomic-embed-text 等），
                                      实现完全离线建库、数据不出域。
    """
    if not texts:
        return []
    if EMBEDDING_BACKEND == "ollama":
        print(f"[Embedding] 本地 Ollama，模型={OLLAMA_EMBED_MODEL}，文本数={len(texts)}",
              file=sys.stderr, flush=True)
        try:
            resp = _get_ollama_embed_client().embeddings.create(
                model=OLLAMA_EMBED_MODEL, input=texts,
            )
            # Ollama 按输入顺序返回，直接取 embedding
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"[Embedding] Ollama 失败，回退云端: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            # 回退到云端，保证建库不中断
    print(f"[Embedding] 调用 API，模型={EMBEDDING_MODEL}，文本数={len(texts)}，base_url={EMBEDDING_API_BASE}",
          file=sys.stderr, flush=True)
    try:
        response = _client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        print(f"[Embedding] API 返回成功，向量数={len(response.data)}", file=sys.stderr, flush=True)
        return [d.embedding for d in response.data]
    except Exception as e:
        print(f"[Embedding] API 调用失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise


def embed_single(text: str) -> list[float]:
    """单个文本转向量"""
    return embed_texts([text])[0]
