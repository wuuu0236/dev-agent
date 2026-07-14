"""
混合检索器 —— BM25（关键词）+ 稠密向量（语义）+ RRF 融合 + Cross-Encoder 重排序

为什么需要混合检索：
  纯向量检索（语义）能理解"怎么读文件"="文件操作"，但可能漏掉精确匹配"open()"的文档
  BM25（关键词）擅长精确匹配，但不懂同义词
  两者互补 → RRF 融合 → 取各自优势

面试一句话：
  "BM25 做关键词召回，稠密向量做语义召回，
   RRF 把两边排名融合，最后 Cross-Encoder 精排。
   比单一方案召回率提升明显。"

运行方式: python -m src.tools.hybrid_retriever
"""

import sys
import io
import os
# 修复 Windows GBK 编码下 emoji 打印报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# HuggingFace 连不上，强制离线模式（模型已在本地缓存）
os.environ['HF_HUB_OFFLINE'] = '1'

import re
import math
import numpy as np
from collections import defaultdict
from sentence_transformers import CrossEncoder


# ================================================================
# Part 1: BM25 检索器（关键词匹配）
# ================================================================

class SimpleBM25:
    """
    最简 BM25 实现 —— 理解 Elasticsearch 默认算法在做什么

    BM25 公式（简化理解）：
      分数 = IDF(词) × TF(词, 文档) 的变体

      其中：
        IDF = log((N - df + 0.5) / (df + 0.5) + 1)
              ↑ 词在所有文档中出现越少，IDF 越高（"稀有词更值钱"）

        TF 变体 = (k1 + 1) × tf / (k1 × (1 - b + b × doc_len/avg_len) + tf)
                  ↑ 词出现次数多分数高，但有上限（防止重复堆砌）
                  ↑ 文档越长，TF 权重越低（长文档天然词多，要惩罚）

    面试不需要背公式，能说清楚：
      "BM25 是 TF-IDF 的改进版，加入了文档长度归一化，
       比 TF-IDF 更合理，Elasticsearch 的默认算法就是这个。"
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        k1: TF 饱和参数，越大 TF 影响越大（一般 1.2~2.0）
        b:  文档长度归一化强度，0=不归一化，1=完全归一化（一般 0.75）
        """
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self._doc_tokens: list[list[str]] = []  # 每个文档的分词结果
        self._idf: dict[str, float] = {}        # 每个词的 IDF
        self._avg_doc_len: float = 0
        self._N: int = 0                        # 文档总数

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        jieba 中文分词 + 英文保留
        切换为 jieba 后，BM25 的词语级匹配精度大幅提升。
        """
        try:
            import jieba
            return [t for t in jieba.cut(text) if len(t.strip()) > 0]
        except ImportError:
            # 降级：按字切分
            return list(text)

    def index(self, documents: list[str]):
        """建索引：分词 → 统计 → 算 IDF"""
        self.documents = documents
        self._N = len(documents)
        self._doc_tokens = [self.tokenize(doc) for doc in documents]

        # 计算平均文档长度
        self._avg_doc_len = sum(len(tokens) for tokens in self._doc_tokens) / max(self._N, 1)

        # 计算 IDF：统计每个词出现在多少篇文档里
        df = defaultdict(int)  # document frequency
        for tokens in self._doc_tokens:
            for word in set(tokens):  # set——同一篇文档里重复出现只算一次
                df[word] += 1

        # IDF 公式
        self._idf = {}
        for word, doc_count in df.items():
            self._idf[word] = math.log(
                (self._N - doc_count + 0.5) / (doc_count + 0.5) + 1
            )

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """搜索，返回 [(文档, BM25分数), ...]"""
        if self._N == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self._doc_tokens):
            score = 0.0
            doc_len = len(doc_tokens)

            for word in query_tokens:
                if word not in self._idf:
                    continue

                idf = self._idf[word]
                tf = doc_tokens.count(word)  # 该词在此文档中出现了几次

                # BM25 的 TF 部分
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                score += idf * numerator / denominator

            scores.append((i, score))

        # 按分数降序排，取 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scores[:top_k] if score > 0]


# ================================================================
# Part 2: RRF（Reciprocal Rank Fusion）融合
# ================================================================

def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    RRF 倒数排名融合 —— 不关心原始分数，只看排名

    公式: score(d) = Σ 1 / (k + rank_i(d))

    k=60 是业界惯例（论文里的经验值），作用：
      - k 越大，排名靠后的文档分数差距越小
      - k=60 意味着：第 1 名得 1/61 ≈ 0.016，第 10 名得 1/70 ≈ 0.014
      - 一个排第 1 + 一个排第 10，总分 > 两个都排第 3

    为什么不用原始分数直接加权？
      BM25 分数范围 [0, ~50]，向量相似度范围 [-1, 1]
      量纲不同，直接加权等于让一个主导另一个
      RRF 只看排名，无视量纲差异 —— 干净
    """
    scores: dict[str, float] = {}
    doc_index: dict[str, str] = {}  # 用于去重：文档内容 → 归一化 key

    def _normalize(text: str) -> str:
        return text.strip()

    # BM25 贡献排名分数
    for rank, (doc, _) in enumerate(bm25_results, start=1):
        key = _normalize(doc)
        doc_index[key] = doc
        scores[key] = scores.get(key, 0) + 1 / (k + rank)

    # 稠密向量贡献排名分数
    for rank, (doc, _) in enumerate(dense_results, start=1):
        key = _normalize(doc)
        doc_index[key] = doc
        scores[key] = scores.get(key, 0) + 1 / (k + rank)

    # 按 RRF 分数降序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_index[key], score) for key, score in sorted_docs]


# ================================================================
# Part 3: Cross-Encoder 重排序
# ================================================================

class SimpleReranker:
    """
    用 Cross-Encoder 精排候选文档

    为什么需要 Rerank？
      BM25 + 稠密向量都是「双塔模型」—— 查询和文档分别编码，速度快但交互浅
      Cross-Encoder 把「查询+文档」拼一起输入，逐字交互 → 更准但更慢
      所以：用快速方法召回 Top-20，再用 Cross-Encoder 精排选 Top-3

    面试时能说清楚：
      "Bi-Encoder（双塔）做召回，速度快；
       Cross-Encoder（交叉编码器）做精排，精度高。
       工业界标准做法就是这个。"
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"📥 加载 Reranker 模型: {model_name} ...")
        self.model = CrossEncoder(model_name, local_files_only=True)
        # 首次调用会很慢（下载模型 ~80MB），之后会缓存

    def rerank(
        self, query: str, candidates: list[str], top_k: int = 3
    ) -> list[tuple[str, float]]:
        """对候选文档重排序，返回 Top-K"""
        if not candidates:
            return []

        # Cross-Encoder 输入格式：(query, document) 对
        pairs = [[query, doc] for doc in candidates]
        scores = self.model.predict(pairs)

        # 按分排序
        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]


# ================================================================
# Part 4: 混合检索器 —— 把上面三个部件串起来
# ================================================================

class HybridRetriever:
    """
    混合检索器：BM25 + 稠密向量 + RRF + Cross-Encoder Rerank

    使用方式：
        retriever = HybridRetriever()
        retriever.index(["文档1", "文档2", ...])       # 建索引
        results = retriever.search("查询", top_k=3)     # 检索
    """

    def __init__(
        self,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_rerank: bool = True,
    ):
        self.enable_rerank = enable_rerank

        # BM25 关键词检索引擎
        self.bm25 = SimpleBM25()

        # 稠密向量：走 Embedding API，不下载本地模型
        print("📥 使用 Embedding API 做向量化...")
        from src.embeddings import embed_texts, embed_single
        self._embed_texts = embed_texts
        self._embed_single = embed_single
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None

        # Reranker（延迟加载——不搜索就不加载，省内存）
        self._reranker: SimpleReranker | None = None
        self._reranker_model = reranker_model

    @property
    def reranker(self) -> SimpleReranker:
        if self._reranker is None:
            self._reranker = SimpleReranker(self._reranker_model)
        return self._reranker

    def index(self, documents: list[str]):
        """建索引：同时建 BM25 和向量两个索引"""
        print(f"🔧 正在索引 {len(documents)} 篇文档...")

        self._documents = documents

        # 1. BM25 索引
        self.bm25.index(documents)
        print(f"   ✅ BM25 索引完成（{len(self.bm25._idf)} 个词）")

        # 2. 稠密向量索引（走 API）
        self._embeddings = np.array(self._embed_texts(documents))
        print(f"   ✅ 向量索引完成（维度: {self._embeddings.shape[1]}）")

    def _dense_search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """稠密向量检索"""
        if self._embeddings is None:
            return []

        query_vec = self._embed_single(query)

        # 余弦相似度
        similarities = np.dot(self._embeddings, query_vec) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (self._documents[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0.3
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 10,
        dense_candidates: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        混合检索主流程

        参数：
          top_k: 最终返回几条
          bm25_candidates: BM25 召回数量（多召一些给 RRF 和 Rerank）
          dense_candidates: 稠密向量召回数量
          rrf_k: RRF 的 k 参数
        """
        # Step 1: 双路召回
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)
        dense_results = self._dense_search(query, top_k=dense_candidates)

        # Step 2: RRF 融合
        fused = reciprocal_rank_fusion(bm25_results, dense_results, k=rrf_k)

        # Step 3: Rerank（可选）
        if self.enable_rerank and len(fused) > top_k:
            # 取 RRF Top-N 送 Reranker（省计算）
            rerank_candidates = [doc for doc, _ in fused[:max(top_k * 3, 10)]]
            reranked = self.reranker.rerank(query, rerank_candidates, top_k=top_k)
            return [
                {"document": doc, "score": float(score), "method": "hybrid+rerank"}
                for doc, score in reranked
            ]
        else:
            # 不用 Rerank，直接返回 RRF 结果
            return [
                {"document": doc, "score": float(score), "method": "hybrid(rrf)"}
                for doc, score in fused[:top_k]
            ]

    def compare(self, query: str, top_k: int = 3) -> dict:
        """
        对比三种方法的结果 —— 用于 debug 和面试展示
        返回 BM25-only、Dense-only、Hybrid+Rerank 三列
        """
        bm25 = self.bm25.search(query, top_k=top_k)
        dense = self._dense_search(query, top_k=top_k)
        hybrid = self.search(query, top_k=top_k)

        return {
            "query": query,
            "bm25_only": [{"doc": d, "score": round(s, 4)} for d, s in bm25],
            "dense_only": [{"doc": d, "score": round(s, 4)} for d, s in dense],
            "hybrid_rerank": hybrid,
        }


# ================================================================
# Part 5: 全局单例 + 便捷函数（和 rag_tool.py 保持一致的接口）
# ================================================================

_retriever: HybridRetriever | None = None
_all_docs: list[str] = []  # 增量模式：记录所有已添加的文档


def get_hybrid_retriever(enable_rerank: bool = True) -> HybridRetriever:
    """全局单例——避免重复加载模型"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(enable_rerank=enable_rerank)
    return _retriever


def hybrid_index(documents: list[str]):
    """便捷函数：建索引"""
    retriever = get_hybrid_retriever()
    retriever.index(documents)


def hybrid_search(query: str, top_k: int = 3) -> str:
    """便捷函数：搜索（返回格式化文本，兼容现有 Agent 工具接口）"""
    retriever = get_hybrid_retriever()
    results = retriever.search(query, top_k=top_k)

    if not results:
        return "没有找到相关内容"

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] [{r['method']}] (相关度: {r['score']:.3f}) {r['document']}")

    return "\n".join(lines)


# ================================================================
# 兼容 rag_tool 接口的函数（Agent 原来调这些名字）
# ================================================================

def search_knowledge(query: str) -> str:
    """搜索知识库（双路：BM25 + 稠密向量 + RRF）——替换原来的纯稠密检索"""
    global _all_docs
    if not _all_docs:
        return "知识库为空。请先用 add_knowledge 添加文档。"
    return hybrid_search(query)


def add_knowledge(texts: list[str]) -> str:
    """增量添加文档到知识库"""
    global _all_docs
    retriever = get_hybrid_retriever()
    _all_docs.extend(texts)
    retriever.index(_all_docs)  # 全量重建索引（项目规模小，够用）
    return f"已添加 {len(texts)} 篇文档到知识库，共 {len(_all_docs)} 条"


def load_file_to_knowledge(filepath: str) -> str:
    """把文件内容加载到知识库"""
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


# ================================================================
# 自测 —— 运行 `python hybrid_retriever.py` 看效果
# ================================================================

if __name__ == "__main__":
    # 测试文档集：模拟知识库场景
    docs = [
        "Python 的文件操作使用 open() 函数，可以指定读写模式如 'r' 和 'w'",
        "LangGraph 是一个用于构建有状态 Agent 的框架，基于有向图",
        "读取文件时建议使用 with open() as f 语法，它会自动关闭文件",
        "Agent 的核心是 ReAct 模式：思考 → 行动 → 观察 → 重复",
        "FastAPI 是一个现代 Python Web 框架，支持异步处理和自动生成 API 文档",
        "Docker 容器化可以将应用和依赖打包在一起，实现环境一致性",
        "MCP（Model Context Protocol）允许 LLM 调用外部工具和服务",
        "向量检索使用余弦相似度来衡量两个文本的语义相关性",
        "BM25 是 Elasticsearch 的默认排序算法，基于 TF-IDF 改进",
        "RAG（检索增强生成）先检索相关文档，再把文档作为上下文给 LLM 生成答案",
    ]

    print("=" * 60)
    print("混合检索器测试")
    print("=" * 60)

    # 不使用 Rerank（首次测试快一点）
    retriever = HybridRetriever(enable_rerank=False)
    retriever.index(docs)

    queries = [
        "怎么用 Python 读文件",
        "什么是 Agent",
        "Docker 有什么用",
    ]

    for q in queries:
        print(f"\n{'─' * 50}")
        comparison = retriever.compare(q, top_k=3)

        print(f"🔍 查询: {q}")
        print(f"\n📋 BM25 (关键词) 结果:")
        for r in comparison["bm25_only"]:
            print(f"   [{r['score']:.4f}] {r['doc'][:60]}")

        print(f"\n📋 稠密向量 (语义) 结果:")
        for r in comparison["dense_only"]:
            print(f"   [{r['score']:.4f}] {r['doc'][:60]}")

        print(f"\n📋 混合 (RRF融合) 结果:")
        for r in comparison["hybrid_rerank"]:
            print(f"   [{r['score']:.4f}] {r['document'][:60]}")

    print(f"\n{'=' * 60}")
    print("✅ 测试完成！对比 BM25 vs 稠密 vs 混合的结果差异")
    print("=" * 60)
