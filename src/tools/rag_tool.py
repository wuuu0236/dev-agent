"""
RAG 工具 —— 让 Agent 能搜索知识库
对照 MaxKB：把「上传文档 → 向量检索 → 返回内容」写成代码
"""

import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


class SimpleVectorStore:
    """
    最简向量数据库 —— 理解 Chroma/Milvus 在做什么
    只用了 NumPy，没有额外依赖

    面试时能说清楚：
      "Chroma 本质上就是存向量 + 做余弦相似度搜索，
       我自己实现过一个简化版来理解它的原理。"
    """

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        print(f"📥 加载向量模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, documents: list[str]):
        """添加文档（对照 MaxKB 的上传文档 → 向量化 → 入库）"""
        print(f"🔧 正在向量化 {len(documents)} 篇文档...")
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        print(f"✅ 向量库就绪，共 {len(documents)} 条")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        检索（对照 MaxKB 的用户提问 → 向量化 → 相似度搜索）
        用余弦相似度，不需要懂算法，只需要知道 cos_sim 越接近 1 越相关
        """
        if self.embeddings is None:
            return ["向量库为空，请先添加文档"]

        query_vec = self.model.encode([query])[0]

        # 余弦相似度（NumPy 一行搞定）
        similarities = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )

        # 取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            if score < 0.3:  # 相似度太低，跳过
                continue
            results.append(f"[相关度: {score:.2f}] {self.documents[idx]}")

        return results if results else ["没有找到相关内容"]

    def save(self, path: str):
        """持久化——下次启动不用重新向量化"""
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)

    def load(self, path: str):
        """从磁盘加载已存在的向量库"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]

    @property
    def is_loaded(self) -> bool:
        return self.embeddings is not None


# ================================================================
# 全局单例 —— 避免每次提问都重新加载模型（几百 MB）
# ================================================================

_store: SimpleVectorStore | None = None
VECTOR_PATH = "./dev_agent_knowledge.pkl"


def get_vector_store(reset: bool = False) -> SimpleVectorStore:
    """获取或创建向量库（单例模式）"""
    global _store

    if _store is not None and not reset:
        return _store

    _store = SimpleVectorStore()

    if os.path.exists(VECTOR_PATH) and not reset:
        print(f"📂 从文件加载已有向量库: {VECTOR_PATH}")
        _store.load(VECTOR_PATH)
        print(f"✅ 加载完成，{len(_store.documents)} 条记录")

    return _store


def add_knowledge(texts: list[str]):
    """向知识库添加文档"""
    store = get_vector_store()
    store.add_documents(texts)
    store.save(VECTOR_PATH)
    return f"已添加 {len(texts)} 篇文档到知识库，共 {len(store.documents)} 条"


def search_knowledge(query: str) -> str:
    """搜索知识库"""
    store = get_vector_store()

    if not store.is_loaded:
        return "知识库为空。请先用 add_knowledge 添加文档。"

    results = store.search(query, top_k=3)
    return "\n".join(results)


def load_file_to_knowledge(filepath: str) -> str:
    """把文件内容加载到知识库（对照 MaxKB 的上传文档功能）"""
    if not os.path.exists(filepath):
        return f"文件 '{filepath}' 不存在"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 按段落切分
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 20]

        if not paragraphs:
            return f"文件 '{filepath}' 没有足够的内容（每个段落至少 20 字）"

        return add_knowledge(paragraphs)

    except UnicodeDecodeError:
        return f"文件 '{filepath}' 不是文本文件"
    except Exception as e:
        return f"加载失败: {str(e)}"
