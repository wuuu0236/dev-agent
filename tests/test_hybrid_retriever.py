"""确定性测试：混合检索的融合层（BM25 + 向量 + RRF）。

为什么测这里：`HybridRetriever.search()` 是线上检索的全部逻辑，而它此前
**零直接测试**（工程审查记录在案的覆盖错位）。这里用假数据把不依赖
Chroma / 不调 LLM 的部分钉死：
  · 分词噪声过滤（标点 / 停用词 / 大小写归一）
  · BM25 零分块不得进候选池
  · 双路命中同一块时 RRF 分数要相加（这是 RRF 的核心语义）
  · 融合后必须截断回候选池大小（否则多开一路召回会连带精排账单上涨）
  · BM25_WEIGHT=0 时要短路，连索引都不建

全部 monkeypatch，不碰真实向量库、不调网络。
"""
import pytest

import src.hybrid_retriever as hr
from src.hybrid_retriever import HybridRetriever, _tokenize


# --- 假数据 ---

class _FakeBM25:
    """按「预先给定的每个 chunk 的分数表」返回分数（键 = chunk 下标）。"""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def get_scores(self, tokens):
        self.calls += 1
        return list(self.scores)


def _chunk(source, content, page=1):
    return {"content": content, "source": source, "page": page, "type": "text"}


@pytest.fixture
def capture(monkeypatch):
    """把检索三件套替换成假实现，并捕获送进精排的候选池。"""
    from src import vector_store, reranker

    box = {
        "vector": [], "bm25": None, "chunks": [], "pool": None,
        "real_get_bm25": hr._get_bm25,   # 短路测试要用真实现
    }

    def fake_search_similar(kb_id, query, top_k=5):
        return [dict(c) for c in box["vector"][:top_k]]

    def fake_get_bm25(kb_id):
        if box["bm25"] is None:
            return None, []
        return box["bm25"], box["chunks"]

    def spy_rerank(query, candidates, top_k, enabled=None):
        box["pool"] = list(candidates)
        return candidates[:top_k]

    monkeypatch.setattr(vector_store, "search_similar", fake_search_similar)
    monkeypatch.setattr(reranker, "rerank", spy_rerank)
    monkeypatch.setattr(hr, "_get_bm25", fake_get_bm25)
    # 默认给双路同权，单项测试可自行覆盖
    monkeypatch.setattr(hr, "BM25_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "VECTOR_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "RETRIEVE_CANDIDATES", 5)
    return box


# --- 分词：噪声过滤 ---

def test_punctuation_dropped():
    """jieba 会把标点也切出来，标点对检索毫无信息，必须滤掉。"""
    toks = _tokenize("混合检索，是怎么做的？")
    assert "，" not in toks
    assert "？" not in toks


def test_stopwords_dropped():
    """高频虚词挤出查询词表，让实词占比更高。"""
    toks = _tokenize("这个是什么的 RAG")
    for w in ("这个", "的"):
        assert w not in toks
    assert "rag" in toks


def test_interrogatives_kept():
    """疑问词刻意保留（标记"问法"而非内容，收进来容易误伤）。"""
    toks = _tokenize("为什么需要 RAG")
    assert "为什么" in toks


def test_lowercased_for_lexical_match():
    """BM25 是词面匹配，大小写不该被当成不同词。"""
    assert "dockerfile" in _tokenize("Dockerfile")
    assert _tokenize("Dockerfile") == _tokenize("dockerfile")


def test_digits_kept_and_bare_underscore_dropped():
    """纯数字（基金代码、版本号）是有意义的检索词，必须保留；
    但 jieba 会把 hybrid_retriever 切成 hybrid / _ / retriever，
    那个孤立下划线没有任何检索信息，要丢掉。
    拆成两半不影响匹配——文档侧和查询侧用的是同一套分词规则。"""
    toks = _tokenize("025490 和 hybrid_retriever")
    assert "025490" in toks
    assert "hybrid" in toks and "retriever" in toks
    assert "_" not in toks


def test_empty_and_pure_punctuation():
    assert _tokenize("") == []
    assert _tokenize("，。！？ \n") == []


# --- BM25 候选：零分块不得进池 ---

def test_zero_score_chunks_excluded(capture):
    """BM25 对"一个词都没对上"的块也返回 0 分；它们没有任何新信息，
    却会让精排多送一批文档（精排按条计费）。必须挡在池外。"""
    capture["vector"] = [_chunk("a.md", "向量独有")]
    capture["chunks"] = [
        _chunk("hit.md", "词面命中"),
        _chunk("zero1.md", "毫无重叠"),
        _chunk("zero2.md", "也毫无重叠"),
    ]
    capture["bm25"] = _FakeBM25([9.9, 0.0, 0.0])

    retriever = HybridRetriever("kb")
    retriever.search("命中", top_k=5)

    sources = [c["source"] for c in capture["pool"]]
    assert "hit.md" in sources
    assert "zero1.md" not in sources
    assert "zero2.md" not in sources


def test_bm25_short_circuits_when_weight_zero(capture, monkeypatch):
    """BM25_WEIGHT=0 时不应去建索引（建索引要拉全库 + 逐条分词，是白算）。

    这里调的是**真实现**：短路分支必须在碰 chromadb 之前就返回，
    否则没装 chromadb 的环境也会被拖挂。
    """
    monkeypatch.setattr(hr, "BM25_WEIGHT", 0.0)
    assert capture["real_get_bm25"]("kb") == (None, [])


def test_bm25_index_is_built_when_weight_on(capture, monkeypatch):
    """权重非 0 时要真的去建索引（不短路）——与上一条构成一对。"""
    from src import vector_store

    monkeypatch.setattr(hr, "BM25_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "_bm25_cache", {})
    called = {"n": 0}

    def fake_count(kb_id):
        called["n"] += 1
        return 0

    monkeypatch.setattr(vector_store, "collection_count", fake_count)
    monkeypatch.setattr(vector_store, "get_all_chunks", lambda kb_id: [])

    assert capture["real_get_bm25"]("kb") == (None, [])
    assert called["n"] == 1, "权重非 0 时应走到 collection_count（说明没有短路）"


# --- RRF 融合语义 ---

def test_rrf_sums_both_routes(capture):
    """两路都命中的块，RRF 分数应是两路贡献之和——这是 RRF 的核心语义：
    多路共识的候选排名应当被抬高。"""
    capture["vector"] = [_chunk("both.md", "两路都命中"), _chunk("vonly.md", "只有向量")]
    capture["chunks"] = [_chunk("both.md", "两路都命中")]
    capture["bm25"] = _FakeBM25([1.0])

    retriever = HybridRetriever("kb")
    retriever.search("查询", top_k=5)

    by_source = {c["source"]: c["rrf_score"] for c in capture["pool"]}
    k = 60
    expected_both = 1.0 / (k + 1) + 1.0 / (k + 1)   # 向量 rank1 + BM25 rank1
    expected_vonly = 1.0 / (k + 2)
    assert by_source["both.md"] == pytest.approx(expected_both)
    assert by_source["vonly.md"] == pytest.approx(expected_vonly)
    assert by_source["both.md"] > by_source["vonly.md"]


def test_pool_truncated_to_candidate_k(capture, monkeypatch):
    """融合后必须截断回候选池大小——否则每加一路召回，精排的账单就跟着涨。
    截断发生在融合之后，所以多路共识的候选会顶掉单路低分候选。"""
    monkeypatch.setattr(hr, "RETRIEVE_CANDIDATES", 3)
    capture["vector"] = [_chunk(f"v{i}.md", f"向量内容{i}") for i in range(5)]
    capture["chunks"] = [_chunk(f"b{i}.md", f"关键词内容{i}") for i in range(5)]
    capture["bm25"] = _FakeBM25([5.0, 4.0, 3.0, 2.0, 1.0])

    retriever = HybridRetriever("kb")
    retriever.search("查询", top_k=3)

    assert len(capture["pool"]) == 3, "候选池必须被截断到 candidate_k"
    # 池内排序 = RRF 降序
    scores = [c["rrf_score"] for c in capture["pool"]]
    assert scores == sorted(scores, reverse=True)


def test_pool_not_padded_when_sources_are_few(capture, monkeypatch):
    """候选不足时不该凭空补条——池子小就是小。"""
    monkeypatch.setattr(hr, "RETRIEVE_CANDIDATES", 50)
    capture["vector"] = [_chunk("v.md", "内容")]
    capture["chunks"] = [_chunk("b.md", "关键词")]
    capture["bm25"] = _FakeBM25([1.0])

    retriever = HybridRetriever("kb")
    retriever.search("查询", top_k=5)

    assert len(capture["pool"]) == 2


def test_dedup_same_chunk_from_both_routes(capture):
    """同一块被两路搜到只应出现一次（否则同一内容占多个名次）。"""
    capture["vector"] = [_chunk("same.md", "同样的内容")]
    capture["chunks"] = [_chunk("same.md", "同样的内容")]
    capture["bm25"] = _FakeBM25([1.0])

    retriever = HybridRetriever("kb")
    retriever.search("查询", top_k=5)

    assert len(capture["pool"]) == 1
