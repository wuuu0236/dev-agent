"""
Embedding 模型封装

使用 sentence-transformers 本地模型，不需要调 API，免费。
模型选择：paraphrase-multilingual-MiniLM-L12-v2
  - 支持中文和英文
  - 输出 384 维向量
  - 模型体积小（~470MB），CPU 也能跑
  - 首次运行会自动下载

为什么用本地模型而不是 API：
  - 免费，无调用次数限制
  - 离线也能用
  - chunk 数量少时比调 API 更快（省去网络往返）
"""
import os
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

# 国内 HuggingFace 被墙，用镜像。设置一次即可。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 全局单例，整个应用共享一个模型
_model: SentenceTransformer | None = None

# 尝试使用 Streamlit 缓存（跨页面持久化，避免重复加载大模型）
try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load_model_st():
        # 优先本地缓存，没有就自动下载（Streamlit Cloud 上没有缓存）
        try:
            return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            return SentenceTransformer(EMBEDDING_MODEL)

    _use_st_cache = True
except ImportError:
    _use_st_cache = False


def get_embedding_model() -> SentenceTransformer:
    """获取 embedding 模型（懒加载，Streamlit 中跨页面缓存，普通脚本中全局单例）"""
    global _model
    if _model is None:
        if _use_st_cache:
            _model = _load_model_st()
        else:
            _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量将文本转成向量"""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """单个文本转向量"""
    return embed_texts([text])[0]
