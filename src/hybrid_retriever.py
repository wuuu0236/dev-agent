"""
混合检索：BM25（关键词） + 向量（语义） + RRF 融合

为什么需要混合检索：
  - 向量检索擅长语义匹配，但搜专有名词（Dockerfile、requirements.txt）效果差
  - BM25 擅长关键词精确匹配，但搜不到近义词
  - 两者互补，结合后 Recall 和 Precision 都比单独好

RRF（倒数排名融合）：
  对每个文档，计算: score = weight1 / (k + rank1) + weight2 / (k + rank2)
  两个排名独立计算，然后加权融合成最终排名。
  RRF 的好处是不需要知道原始分数的分布（BM25 和向量分数的范围完全不同）。
"""
import jieba
from rank_bm25 import BM25Okapi
from src.vector_store import search_similar, get_all_chunks
from src.config import TOP_K_RETRIEVE, BM25_WEIGHT, VECTOR_WEIGHT


def _tokenize(text: str) -> list[str]:
    """中文分词：用 jieba 做词语级切分。

    为什么不用按字切分：
      "什么是混合检索" → 按字切: ["什","么","是","混","合","检","索"]
      → 搜"检索"只能匹配到"检"和"索"两个字，毫无意义

      "什么是混合检索" → jieba: ["什么","是","混合","检索"]
      → 搜"检索"能精确匹配到"检索"这个词，BM25 才有真正的关键词能力
    """
    return list(jieba.cut(text))


class HybridRetriever:
    """
    混合检索器，每个知识库一个实例。

    使用方式：
      retriever = HybridRetriever(kb_id)
      results = retriever.search("查询内容", top_k=5)
    """

    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.chunks: list[dict] = []      # 所有 chunk
        self.bm25: BM25Okapi | None = None  # BM25 索引

    def _build_bm25(self):
        """构建 BM25 索引（从向量库里取出所有 chunk）"""
        self.chunks = get_all_chunks(self.kb_id)
        if not self.chunks:
            self.bm25 = None
            return
        tokenized = [_tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        BM25 关键词检索
        返回：[(chunk在self.chunks中的索引, 分数), ...]
        """
        if self.bm25 is None:
            return []
        tokenized = _tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        # 取 top_k
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        """
        混合检索主流程：
        1. 向量检索（语义匹配）
        2. BM25 检索（关键词匹配）
        3. RRF 融合两路排名
        4. 返回融合后的 top_k 结果
        """
        # --- 向量检索 ---
        vector_results = search_similar(self.kb_id, query, top_k=top_k * 2)
        # 给每个结果标上向量检索的排名（1 = 最相关）
        for rank, r in enumerate(vector_results, start=1):
            r["vector_rank"] = rank

        # --- BM25 检索 ---
        self._build_bm25()
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        # 把 BM25 结果映射到 chunk 数据
        bm25_mapped = []
        for rank, (idx, score) in enumerate(bm25_results, start=1):
            chunk = self.chunks[idx]
            bm25_mapped.append({
                "content": chunk["content"],
                "source": chunk["source"],
                "page": chunk["page"],
                "bm25_rank": rank,
                "bm25_score": float(score)
            })

        # --- RRF 融合 ---
        # 用 dict 去重（同一内容可能两路都搜到），key = source + content[:50]
        merged = {}
        k = 60  # RRF 公式中的平滑常数

        for r in vector_results:
            key = f"{r['source']}_{r['content'][:50]}"
            merged[key] = {
                "content": r["content"],
                "source": r["source"],
                "page": r.get("page", 0),
                "rrf_score": VECTOR_WEIGHT / (k + r["vector_rank"])
            }

        for r in bm25_mapped:
            key = f"{r['source']}_{r['content'][:50]}"
            bm25_contrib = BM25_WEIGHT / (k + r["bm25_rank"])
            if key in merged:
                merged[key]["rrf_score"] += bm25_contrib
            else:
                merged[key] = {
                    "content": r["content"],
                    "source": r["source"],
                    "page": r.get("page", 0),
                    "rrf_score": bm25_contrib
                }

        # 按 RRF 分数从高到低排序
        sorted_results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:top_k]
# ================================================================
# 模块级函数：供 v4 Agent 和 MCP 调用（统一入口）
# ================================================================

_DEFAULT_KB = "agent"


def _ensure_kb() -> str:
    """获取最近的知识库 ID，没有就创建默认的"""
    from src.database import list_kbs, create_kb, init_db
    init_db()
    kbs = list_kbs()
    if kbs:
        return kbs[0]["id"]
    from src.vector_store import create_collection
    kb = create_kb(_DEFAULT_KB, "Agent 默认知识库")
    create_collection(kb["id"])
    return kb["id"]


def search_knowledge(query: str) -> str:
    """搜索知识库，返回纯文本（供 Agent/MCP 调用）"""
    kb_id = _ensure_kb()
    retriever = HybridRetriever(kb_id)
    results = retriever.search(query, top_k=5)
    if not results:
        return "知识库中未找到相关内容，请先上传文档。"
    lines = []
    for i, r in enumerate(results, 1):
        source = f"[{i}] 来源: {r['source']}"
        if r.get("page"):
            source += f", 第 {r['page']} 页"
        lines.append(source)
        lines.append(r["content"][:300])
        lines.append("")
    return "\n".join(lines)


def add_knowledge(texts: list[str]) -> str:
    """把文本添加到知识库"""
    kb_id = _ensure_kb()
    from src.chunker import chunk_parsed
    docs = [{"text": t, "page": None, "source": "agent_"} for t in texts]
    chunks = chunk_parsed(docs)
    if not chunks:
        return "没有可添加的内容"
    from src.vector_store import add_chunks
    add_chunks(kb_id, chunks)
    return f"已添加 {len(chunks)} 条内容到知识库"


def load_file_to_knowledge(filepath: str) -> str:
    """把文件加载到知识库（含安全检查）"""
    from src.tools.safety import check_path_safety
    safe, reason = check_path_safety(filepath)
    if not safe:
        return f"安全拦截: {reason}"
    from src.parser import parse_file
    from src.chunker import chunk_parsed
    from src.vector_store import add_chunks
    kb_id = _ensure_kb()
    try:
        parsed = parse_file(filepath)
        if not parsed:
            return f"文件 '{filepath}' 没有可解析的内容"
        chunks = chunk_parsed(parsed)
        if not chunks:
            return f"文件 '{filepath}' 没有足够的内容"
        add_chunks(kb_id, chunks)
        return f"已加载文件 '{filepath}'，共 {len(chunks)} 条内容到知识库"
    except Exception as e:
        return f"加载失败: {type(e).__name__}: {e}"