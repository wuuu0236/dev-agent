"""
向量存储：基于 Chroma

每个知识库 = 一个 Chroma Collection，数据隔离。
Collection 命名规则：kb_{kb_id}

存储内容：
  - documents: chunk 文本内容
  - metadatas: {source, page, chunk_index} 用于显示引用
  - ids: chunk_{source}_{chunk_index} 唯一标识
"""
import sys

import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR, EMBEDDING_DIM
from src.embeddings import embed_texts, embed_query


_client = None  # 单例缓存（chromadb 1.x 里 PersistentClient 是工厂函数，不做类型注解）


def _get_client() -> chromadb.PersistentClient:
    """获取 Chroma 客户端（进程内单例，只创建一次）。

    之前每次调用都新建 PersistentClient——chroma 的共享系统在释放旧客户端时
    偶发崩溃（RustBindingsAPI.stop 的 `del self.bindings`）。正确用法是进程内
    单例，Streamlit 多页面重跑也不会重复创建。
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


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


def add_chunks(kb_id: str, chunks: list[dict], replace_source: bool = True):
    """
    将 chunk 列表写入向量库。

    chunks 格式：[{content, source, page, chunk_index}, ...]
    Chroma 会自动调用 embedding 函数把 content 转成向量。

    replace_source=True（默认）：写入前先清掉同来源的旧 chunk，让"重传同一份
    文件"等价于**替换**而不是叠加。必要的原因：chunk id 是 `{source}_chunk{i}`，
    重传修订版时若新文件更短（比如从 10 块变成 6 块），尾部 chunk6~9 的 id
    不会被覆盖，会作为旧内容永远留在库里被检索到。整库重建场景可传 False 省一次查询。

    幂等性保证：向量一定先算完才动旧数据，所以任何一次 embedding 失败都不会
    让已有内容受损（函数会直接抛异常，库里保持原样）。
    """
    if not chunks:
        return

    client = _get_client()
    collection = client.get_collection(_collection_name(kb_id))

    # 先算向量、再删旧块——**顺序不能反**。反过来的话，embedding 调用一旦失败
    # （网络抖动 / 额度用尽 / 超时），旧 chunk 已经删掉、新 chunk 又没写进去，
    # 这份文档就彻底从库里消失了。整库重建会把这个隐患放大成"整个库没了"。
    documents = [c["content"] for c in chunks]
    embeddings = embed_texts(documents)

    if replace_source:
        for src in {c["source"] for c in chunks}:
            delete_chunks_by_source(kb_id, src)

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
    # 知识库内容变化 → 语义缓存作废（等价"表被写入"清查询缓存）
    _clear_query_cache(kb_id)


def search_similar(kb_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    向量检索：在知识库中搜索与 query 最相似的 chunk。

    返回：[{content, source, page, score}, ...]
    score 是余弦相似度，越大越相关。
    """
    client = _get_client()
    collection = client.get_collection(_collection_name(kb_id))

    # 检索查询走 embed_query（自动带 BGE 指令前缀）；文档入库走 embed_texts，不带前缀
    query_embedding = [embed_query(query)]

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


def _clear_query_cache(kb_id: str):
    """知识库内容变化时作废语义缓存。惰性 import，避免模块循环依赖。"""
    try:
        from src.query_cache import clear_kb_cache
        clear_kb_cache(kb_id)
    except Exception:
        pass  # 缓存模块不可用不影响主流程


def delete_collection(kb_id: str):
    """删除知识库对应的向量集合"""
    client = _get_client()
    try:
        client.delete_collection(_collection_name(kb_id))
    except Exception:
        pass
    _clear_query_cache(kb_id)


def delete_chunks_by_source(kb_id: str, source: str) -> int:
    """删除某个来源文件在向量库中的全部 chunk，返回实际删除条数。

    为什么需要它：documents 表（SQLite）记录的只是元数据，chunk 内容存在
    Chroma 里。只删 SQLite 不删 Chroma，会出现"文档从列表消失、内容却仍被
    检索到并被引用"的幽灵引用——用户以为删干净了，答案里却还在引用它。

    为什么按 metadata 的 source 删、而不是按 id 前缀删：chunk id 是
    `{source}_chunk{i}`，看着能前缀匹配，但文件名含特殊字符、或同名文件互相
    覆盖时会失准；source 是入库时写进 metadata 的精确值，过滤最可靠。
    """
    client = _get_client()
    try:
        collection = client.get_collection(_collection_name(kb_id))
    except Exception:
        return 0  # 库不存在 = 没有 chunk 要删，不视为错误
    try:
        got = collection.get(where={"source": source})
        ids = got.get("ids") or []
        if not ids:
            return 0
        collection.delete(ids=ids)
        _clear_query_cache(kb_id)  # 内容变了，语义缓存作废
        return len(ids)
    except Exception as e:
        print(f"[VectorStore] 按来源删除失败 source={source}: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return 0


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
