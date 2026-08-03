"""
向量存储：基于 Chroma

每个知识库 = 一个 Chroma Collection，数据隔离。
Collection 命名规则：kb_{kb_id}

存储内容：
  - documents: chunk 文本内容
  - metadatas: {source, page, chunk_index} 用于显示引用
  - ids: chunk_{source}_{chunk_index} 唯一标识
"""
import shutil
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR, EMBEDDING_DIM
from src.embeddings import embed_texts, embed_single


def _get_client() -> chromadb.PersistentClient:
    """获取 Chroma 客户端（持久化存储）"""
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )


def _collection_name(kb_id: str) -> str:
    """知识库 ID → Chroma Collection 名称"""
    return f"kb_{kb_id}"


def create_collection(kb_id: str):
    """为知识库创建空的向量集合"""
    client = _get_client()
    name = _collection_name(kb_id)
    # 如果已存在就先删掉重建
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.create_collection(name, metadata={"hnsw:space": "cosine"})


def add_chunks(kb_id: str, chunks: list[dict]):
    """
    将 chunk 列表写入向量库。

    chunks 格式：[{content, source, page, chunk_index}, ...]
    Chroma 会自动调用 embedding 函数把 content 转成向量。
    """
    if not chunks:
        return

    client = _get_client()
    collection = client.get_collection(_collection_name(kb_id))

    # 使用 Embedding API 向量化
    documents = [c["content"] for c in chunks]
    embeddings = embed_texts(documents)

    ids = [f"{c['source']}_chunk{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "page": c.get("page") or 0,
            "chunk_index": c["chunk_index"],
            "type": c.get("type", "text"),
            # Chroma 元数据不支持 None，空字符串占位；空串在下游视为无图
            "image": c.get("image") or "",
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_similar(kb_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    向量检索：在知识库中搜索与 query 最相似的 chunk。

    返回：[{content, source, page, score}, ...]
    score 是余弦相似度，越大越相关。
    """
    client = _get_client()
    collection = client.get_collection(_collection_name(kb_id))

    query_embedding = [embed_single(query)]

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return [
        {
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i].get("page", 0),
            "type": results["metadatas"][0][i].get("type", "text"),
            "image": results["metadatas"][0][i].get("image") or None,
            "score": 1 - results["distances"][0][i]  # Chroma 返回距离，转成相似度
        }
        for i in range(len(results["documents"][0]))
    ]


def get_all_chunks(kb_id: str) -> list[dict]:
    """获取知识库中的所有 chunk（用于 BM25 检索和统计）"""
    client = _get_client()
    collection = client.get_collection(_collection_name(kb_id))
    results = collection.get()

    if not results["documents"]:
        return []

    return [
        {
            "content": results["documents"][i],
            "source": results["metadatas"][i]["source"],
            "page": results["metadatas"][i].get("page", 0),
            "type": results["metadatas"][i].get("type", "text"),
            "image": results["metadatas"][i].get("image") or None,
            "chunk_id": results["ids"][i]
        }
        for i in range(len(results["documents"]))
    ]


def delete_collection(kb_id: str):
    """删除知识库对应的向量集合"""
    client = _get_client()
    try:
        client.delete_collection(_collection_name(kb_id))
    except Exception:
        pass


def collection_count(kb_id: str) -> int:
    """知识库中 chunk 的数量"""
    client = _get_client()
    try:
        collection = client.get_collection(_collection_name(kb_id))
        return collection.count()
    except Exception:
        return 0


def check_embedding_dim(kb_id: str) -> tuple[bool, str]:
    """检查知识库 embedding 维度是否与当前配置一致。

    不一致说明这个库是用旧模型建的（如 text2vec 768 维），当前 bge-large 1024 维
    检索时会抛 dimension 错误。返回 (是否兼容, 提示)。
    """
    from src.config import EMBEDDING_DIM
    try:
        collection = _get_client().get_collection(_collection_name(kb_id))
        got = collection.get(limit=1, include=["embeddings"])
        emb = got.get("embeddings")
        if emb is not None and len(emb) > 0:
            dim = len(emb[0])
            if dim == EMBEDDING_DIM:
                return True, ""
            return False, (
                f"该知识库用 {dim} 维 embedding 模型构建，当前模型输出 {EMBEDDING_DIM} 维，"
                f"检索会报维度不匹配。请用当前模型重新上传文档重建此库。"
            )
        return True, ""  # 空库无需检查
    except Exception:
        return True, ""  # 无法检查则放行，让实际检索报错
