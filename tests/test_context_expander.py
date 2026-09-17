"""Parent-Child 邻域扩展（src/context_expander.py + chunk_index 透传）单测。

守的行为（对应 docs/technical-optimization-plan.md §3.6，另加透传回归）：
  · expand_contexts：关闭原样返回；相邻块合并；重叠邻域的第二个命中**保留
    原样**（文档原稿会把它丢掉——丢的是一条带分数的精排结果）；超长截断；
    老索引缺 chunk_index 保底不扩展；命中原文存进 original_content
  · chunk_index 透传：向量路与 BM25 路的候选都要带 chunk_index
    （metadata 里一直存着，此前没带出来——不修这个，扩展永远拿不到定位）
  · get_chunks_by_source_range：字段映射 + 按 chunk_index 升序

全部 monkeypatch，不碰真实 Chroma（CI 约定）。
"""
import pytest

import src.context_expander as ce
import src.vector_store as vs
import src.hybrid_retriever as hr
from src import vector_store
from src.hybrid_retriever import HybridRetriever


# ================================================================
# expand_contexts：邻域扩展
# ================================================================

@pytest.fixture
def expanded_env(monkeypatch):
    """打开扩展开关 + 可编程的假邻域查询，返回 (range_calls, neighbors_by_key)。"""
    monkeypatch.setattr(ce, "CONTEXT_EXPANSION_ENABLED", True)
    range_calls: list[tuple] = []
    neighbors_by_key: dict = {}

    def fake_range(kb_id, source, start, end):
        range_calls.append((kb_id, source, start, end))
        return [dict(c) for c in neighbors_by_key.get((source, start, end), [])]

    monkeypatch.setattr(vector_store, "get_chunks_by_source_range", fake_range)
    return {"range_calls": range_calls, "neighbors": neighbors_by_key}


def _neighbor(source, idx, content):
    return {"content": content, "source": source, "page": 1, "chunk_index": idx}


def test_disabled_returns_unchanged(monkeypatch):
    """开关关闭：原样返回（对应文档测试用例 1）。"""
    monkeypatch.setattr(ce, "CONTEXT_EXPANSION_ENABLED", False)
    hits = [{"content": "原文", "source": "a.md", "chunk_index": 2}]
    out = ce.expand_contexts(hits, "kb")
    assert out == hits


def test_empty_hits_returns_unchanged(expanded_env):
    assert ce.expand_contexts([], "kb") == []


def test_neighbors_merged_and_original_preserved(expanded_env):
    """命中 idx=2，前后各 2 → 拉 0-4 五块合并；original_content 存命中原文。"""
    expanded_env["neighbors"][("a.md", 0, 4)] = [
        _neighbor("a.md", i, f"第{i}块内容") for i in range(5)
    ]
    hit = {"content": "第2块内容", "source": "a.md", "chunk_index": 2,
           "rerank_score": 0.8}
    out = ce.expand_contexts([hit], "kb")

    assert len(out) == 1
    assert out[0]["content"] == "\n".join(f"第{i}块内容" for i in range(5))
    assert out[0]["original_content"] == "第2块内容"
    assert out[0]["rerank_score"] == 0.8  # 其余字段（含分数）原样保留
    assert expanded_env["range_calls"] == [("kb", "a.md", 0, 4)]


def test_merged_text_truncated_to_max_chars(expanded_env, monkeypatch):
    """合并后超过上限 → 截断加省略号（对应文档测试用例 4）。"""
    monkeypatch.setattr(ce, "CONTEXT_EXPAND_MAX_CHARS", 10)
    expanded_env["neighbors"][("a.md", 0, 4)] = [
        _neighbor("a.md", i, "x" * 30) for i in range(5)
    ]
    hit = {"content": "x" * 30, "source": "a.md", "chunk_index": 2}
    out = ce.expand_contexts([hit], "kb")

    assert out[0]["content"] == "x" * 10 + "..."
    assert out[0]["original_content"] == "x" * 30


def test_overlapping_neighbor_keeps_second_hit(expanded_env):
    """同一文档两条相邻命中：第一条扩展，第二条**保留原样**（不重复扩展、
    也不丢弃——文档原稿在这里 continue 丢掉一条精排结果，引用会凭空少一条）。"""
    expanded_env["neighbors"][("a.md", 0, 4)] = [
        _neighbor("a.md", i, f"第{i}块") for i in range(5)
    ]
    hits = [
        {"content": "第2块", "source": "a.md", "chunk_index": 2, "rerank_score": 0.9},
        {"content": "第3块", "source": "a.md", "chunk_index": 3, "rerank_score": 0.8},
    ]
    out = ce.expand_contexts(hits, "kb")

    assert len(out) == 2  # 两条都在，没丢
    assert out[0]["content"].count("第") == 5          # 第一条：邻域合并文本
    assert out[1]["content"] == "第3块"                # 第二条：原样
    assert "original_content" not in out[1]
    assert len(expanded_env["range_calls"]) == 1       # 邻域也只查了一次


def test_different_sources_both_expanded(expanded_env):
    """不同文档的命中各自扩展，互不影响。"""
    expanded_env["neighbors"][("a.md", 0, 4)] = [_neighbor("a.md", i, f"a{i}") for i in range(5)]
    expanded_env["neighbors"][("b.md", 0, 4)] = [_neighbor("b.md", i, f"b{i}") for i in range(5)]
    hits = [
        {"content": "a2", "source": "a.md", "chunk_index": 2},
        {"content": "b2", "source": "b.md", "chunk_index": 2},
    ]
    out = ce.expand_contexts(hits, "kb")
    assert out[0]["content"].startswith("a0")
    assert out[1]["content"].startswith("b0")


def test_hit_without_chunk_index_left_alone(expanded_env):
    """老索引缺 chunk_index：不扩展、原样保留（保底不丢数据、不报错）。"""
    hit = {"content": "原文", "source": "a.md"}  # 没有 chunk_index
    out = ce.expand_contexts([hit], "kb")
    assert out == [hit]
    assert expanded_env["range_calls"] == []


def test_no_neighbors_found_left_alone(expanded_env):
    """范围查询没拉到任何块（如数据不一致）：原样保留。"""
    hit = {"content": "原文", "source": "a.md", "chunk_index": 2}
    out = ce.expand_contexts([hit], "kb")
    assert out == [hit]
    assert out[0]["content"] == "原文"


# ================================================================
# chunk_index 透传：向量路与 BM25 路的候选都要带
# ================================================================

class _FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return list(self.scores)


def _passthrough_env(monkeypatch, vector_chunks, bm25_chunks, bm25_scores):
    """向量 + BM25 双路假环境，返回 (search 结果捕获, rerank 池捕获)。"""
    from src import reranker

    def fake_search_similar(kb_id, query, top_k=5):
        results = [dict(c) for c in vector_chunks][:top_k]
        for rank, r in enumerate(results, start=1):
            r["vector_rank"] = rank
        return results

    def fake_get_bm25(kb_id):
        if bm25_chunks is None:
            return None, []
        return _FakeBM25(bm25_scores), bm25_chunks

    pool_box = {}

    def spy_rerank(query, candidates, top_k, enabled=None):
        pool_box["pool"] = list(candidates)
        return candidates[:top_k]

    monkeypatch.setattr(vector_store, "search_similar", fake_search_similar)
    monkeypatch.setattr(reranker, "rerank", spy_rerank)
    monkeypatch.setattr(hr, "_get_bm25", fake_get_bm25)
    monkeypatch.setattr(hr, "BM25_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "VECTOR_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "RETRIEVE_CANDIDATES", 5)
    return pool_box


def test_vector_path_carries_chunk_index(monkeypatch):
    """向量路：search_similar 的 chunk_index 要一路穿过 RRF 合并活到精排输出。"""
    chunks = [{"content": "甲" * 60, "source": "a.md", "page": 1,
               "type": "text", "chunk_index": 3}]
    pool = _passthrough_env(monkeypatch, chunks, None, None)

    out = HybridRetriever("kb").search("问题", top_k=1)
    assert out[0]["chunk_index"] == 3
    assert pool["pool"][0]["chunk_index"] == 3


def test_bm25_path_carries_chunk_index(monkeypatch):
    """BM25 路：get_all_chunks 的 chunk_index 经 bm25_mapped / RRF 合并不丢失。"""
    chunks = [{"content": "乙" * 60, "source": "b.md", "page": 1,
               "type": "text", "chunk_index": 7}]
    pool = _passthrough_env(monkeypatch, [], chunks, [1.0, 0.0])

    out = HybridRetriever("kb").search("问题", top_k=1)
    assert out[0]["chunk_index"] == 7
    assert pool["pool"][0]["chunk_index"] == 7


# ================================================================
# get_chunks_by_source_range：字段映射 + 排序
# ================================================================

def test_get_chunks_by_source_range_maps_and_sorts(monkeypatch):
    """fake collection.get 返回乱序结果 → 按 chunk_index 升序 + 字段映射。"""
    captured = {}

    class _FakeCollection:
        def get(self, where=None, include=None):
            captured["where"] = where
            captured["include"] = include
            return {
                "ids": ["b_chunk3", "b_chunk1"],
                "documents": ["块3", "块1"],
                "metadatas": [
                    {"source": "b.md", "page": 2, "chunk_index": 3},
                    {"source": "b.md", "page": 1, "chunk_index": 1},
                ],
            }

    class _FakeClient:
        def get_collection(self, name):
            return _FakeCollection()

    monkeypatch.setattr(vs, "_get_client", lambda: _FakeClient())

    out = vs.get_chunks_by_source_range("kb", "b.md", 0, 4)

    # where 过滤条件正确（source + chunk_index 范围），按 chunk_index 升序
    assert captured["where"] == {"$and": [
        {"source": {"$eq": "b.md"}},
        {"chunk_index": {"$gte": 0}},
        {"chunk_index": {"$lte": 4}},
    ]}
    assert [c["chunk_index"] for c in out] == [1, 3]
    assert out[0]["content"] == "块1"
    assert out[0]["source"] == "b.md"
