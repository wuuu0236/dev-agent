"""Rerank 精排测试：排序正确性 + 各种失败下的降级行为。

降级是这里的重点——精排是"锦上添花"的一步，它挂掉绝不能让整个问答挂掉。
"""
import pytest

from src.reranker import rerank

CANDIDATES = [
    {"content": "文档 A", "source": "a.md"},
    {"content": "文档 B", "source": "b.md"},
    {"content": "文档 C", "source": "c.md"},
]


def _fake_scores(mapping):
    """按 content 前缀返回固定分数，模拟 API。"""
    def _f(query, docs):
        return [mapping.get(d[:4], 0.0) for d in docs]
    return _f


def test_empty_candidates():
    assert rerank("q", [], top_k=5) == []


def test_disabled_keeps_original_order(monkeypatch):
    """关闭精排 → 原样返回粗排前 top_k，且不写入 rerank_score。"""
    out = rerank("q", CANDIDATES, top_k=2, enabled=False)
    assert [c["source"] for c in out] == ["a.md", "b.md"]
    assert "rerank_score" not in out[0]


def test_reorders_by_cross_encoder_score(monkeypatch):
    """精排按分数重排——粗排排第一的 A 被压到最后。"""
    monkeypatch.setattr("src.reranker._score_batch",
                        _fake_scores({"文档 A": 0.1, "文档 B": 0.9, "文档 C": 0.5}))
    out = rerank("q", CANDIDATES, top_k=3, enabled=True)
    assert [c["source"] for c in out] == ["b.md", "c.md", "a.md"]
    assert out[0]["rerank_score"] == 0.9


def test_truncates_to_top_k(monkeypatch):
    monkeypatch.setattr("src.reranker._score_batch", _fake_scores({"文档 A": 0.1, "文档 B": 0.9, "文档 C": 0.5}))
    out = rerank("q", CANDIDATES, top_k=2, enabled=True)
    assert len(out) == 2
    assert [c["source"] for c in out] == ["b.md", "c.md"]


def test_original_fields_preserved(monkeypatch):
    """精排只加分数、不改内容——下游引用/页码依赖这些字段。"""
    monkeypatch.setattr("src.reranker._score_batch", _fake_scores({"文档 A": 0.9}))
    cands = [{"content": "文档 A", "source": "a.md", "page": 7, "rrf_score": 0.016}]
    out = rerank("q", cands, top_k=1, enabled=True)
    assert out[0]["page"] == 7
    assert out[0]["rrf_score"] == 0.016
    assert out[0]["rerank_score"] == 0.9


def test_api_failure_degrades_to_rough_order(monkeypatch):
    """API 失败 → 降级为粗排顺序，不抛异常。"""
    monkeypatch.setattr("src.reranker._score_batch", lambda q, d: None)
    out = rerank("q", CANDIDATES, top_k=2, enabled=True)
    assert [c["source"] for c in out] == ["a.md", "b.md"]


def test_missing_api_key_degrades(monkeypatch):
    monkeypatch.setattr("src.reranker.RERANK_API_KEY", "")
    monkeypatch.setattr("src.reranker._score_batch",
                        lambda q, d: pytest.fail("没有 key 时不应发起请求"))
    out = rerank("q", CANDIDATES, top_k=2, enabled=True)
    assert [c["source"] for c in out] == ["a.md", "b.md"]


def test_batching_splits_large_candidate_pool(monkeypatch):
    """40 条候选 / 每批 32 → 分 2 批（32 + 8）。"""
    calls = []

    def fake(query, docs):
        calls.append(len(docs))
        return [0.5] * len(docs)

    monkeypatch.setattr("src.reranker._score_batch", fake)
    cands = [{"content": f"d{i}", "source": f"s{i}"} for i in range(40)]
    rerank("q", cands, top_k=5, enabled=True)
    assert calls == [32, 8]


def test_midway_batch_failure_degrades_whole_call(monkeypatch):
    """第二批失败 → 整体降级（保证排序一致，不出现"前 32 条精排过、后 8 条没有"）。"""
    state = {"n": 0}

    def fake(query, docs):
        state["n"] += 1
        return None if state["n"] == 2 else [0.5] * len(docs)

    monkeypatch.setattr("src.reranker._score_batch", fake)
    cands = [{"content": f"d{i}", "source": f"s{i}"} for i in range(40)]
    out = rerank("q", cands, top_k=5, enabled=True)
    assert [c["source"] for c in out] == ["s0", "s1", "s2", "s3", "s4"]


def test_long_document_is_truncated_before_sending(monkeypatch):
    """超长文本截断后再送审，防止请求体爆炸。"""
    seen = {}

    def fake(query, docs):
        seen["docs"] = docs
        return [0.5] * len(docs)

    monkeypatch.setattr("src.reranker._score_batch", fake)
    monkeypatch.setattr("src.reranker.RERANK_MAX_DOC_CHARS", 10)
    rerank("q", [{"content": "超" * 100, "source": "a.md"}], top_k=1, enabled=True)
    assert len(seen["docs"][0]) == 10


def test_scores_restored_to_input_order(monkeypatch):
    """API 按分数降序返回 [{index, score}]，必须还原到输入顺序再排序，否则错位。"""
    def fake(query, docs):
        # 故意乱序返回：说 docs[2] 最相关
        return [0.1, 0.2, 0.95]

    monkeypatch.setattr("src.reranker._score_batch", fake)
    out = rerank("q", CANDIDATES, top_k=3, enabled=True)
    assert out[0]["source"] == "c.md"
    assert out[0]["rerank_score"] == 0.95
