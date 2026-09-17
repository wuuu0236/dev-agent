"""Multi-Query 查询扩展（src/query_expand.py + HybridRetriever.search_multi）单测。

守的行为（对应 docs/technical-optimization-plan.md §4.5，检索侧按文档 §4.1
"合并去重后送精排"的原则实现为 search_multi）：
  · expand_query：关闭 / 失败退回 [query]；原始 query 打头；保序去重；
    截断到 QUERY_EXPANSION_COUNT
  · search_multi：跨查询共识在精排之前融合（RRF 跨查询累加）；
    **精排只做一次**且用主查询打分；融合后截断回候选池

全部 monkeypatch，不碰真实向量库、不调网络（CI 约定）。
"""
import pytest

import src.hybrid_retriever as hr
import src.query_expand as qe
from src.hybrid_retriever import HybridRetriever
from src import vector_store  # 惰性 import 发生在函数内，patch 必须打到源模块


# ================================================================
# expand_query：查询扩展
# ================================================================

def _make_client(reply):
    """假 OpenAI 客户端：reply 为 str 时返回按行拆分前的文本，为 Exception 时抛出。"""
    calls: list[dict] = []

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if isinstance(reply, Exception):
                raise reply

            class _Msg:
                content = reply

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _FakeClient:
        chat = type("C", (), {})()
        chat.completions = _FakeCompletions()

    return _FakeClient(), calls


def test_disabled_returns_original_without_llm(monkeypatch):
    """开关关闭：原样返回 [query]，且根本不该构造/调用 LLM 客户端。"""
    monkeypatch.setattr(qe, "QUERY_EXPANSION_ENABLED", False)

    def _boom():
        raise AssertionError("关闭时不应构造/调用 LLM 客户端")

    monkeypatch.setattr(qe, "_get_client", _boom)
    assert qe.expand_query("混合检索是怎么做的") == ["混合检索是怎么做的"]


def test_expansion_keeps_original_first(enabled_count4, monkeypatch):
    """LLM 返回 3 行扩展：结果 4 条、原始 query 打头（对应文档测试用例 2）。"""
    client, calls = _make_client("BM25 关键词召回怎么实现\n混合检索里的 BM25 部分怎么做\n关键词检索的实现方式")
    monkeypatch.setattr(qe, "_get_client", lambda: client)

    out = qe.expand_query("BM25 混合检索是怎么做的")
    assert out[0] == "BM25 混合检索是怎么做的"
    assert len(out) == 4
    assert len(set(out)) == 4  # 无重复

    # 送审结构：system 提示词里带了扩展数量要求，user 是原始问题
    assert "4 个" in calls[0]["messages"][0]["content"]
    assert calls[0]["messages"][1]["content"] == "BM25 混合检索是怎么做的"


@pytest.fixture
def enabled_count4(monkeypatch):
    monkeypatch.setattr(qe, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(qe, "QUERY_EXPANSION_COUNT", 4)


def test_expansion_truncated_to_count(monkeypatch):
    """超过 QUERY_EXPANSION_COUNT 的扩展行被截掉（count=3 时 3 行输入 → 3 条输出）。"""
    monkeypatch.setattr(qe, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(qe, "QUERY_EXPANSION_COUNT", 3)
    client, _ = _make_client("角度一\n角度二\n角度三")
    monkeypatch.setattr(qe, "_get_client", lambda: client)

    out = qe.expand_query("原始问题")
    assert out == ["原始问题", "角度一", "角度二"]


def test_llm_exception_falls_back_to_original(monkeypatch):
    """LLM 抛异常：退回 [query]，不向上抛——扩展失败不能拖垮问答。"""
    monkeypatch.setattr(qe, "QUERY_EXPANSION_ENABLED", True)
    client, _ = _make_client(RuntimeError("API 限流"))
    monkeypatch.setattr(qe, "_get_client", lambda: client)

    assert qe.expand_query("原始问题") == ["原始问题"]


def test_duplicate_of_original_dropped(enabled_count4, monkeypatch):
    """LLM 复读了原始问题：去重后原始 query 只出现一次。"""
    client, _ = _make_client("原始问题\n角度二\n角度三")
    monkeypatch.setattr(qe, "_get_client", lambda: client)

    out = qe.expand_query("原始问题")
    assert out == ["原始问题", "角度二", "角度三"]


def test_blank_lines_filtered(enabled_count4, monkeypatch):
    """空行被过滤，不占扩展名额。"""
    client, _ = _make_client("\n角度二\n\n  \n角度三\n")
    monkeypatch.setattr(qe, "_get_client", lambda: client)

    out = qe.expand_query("原始问题")
    assert out == ["原始问题", "角度二", "角度三"]


# ================================================================
# HybridRetriever.search_multi：多查询联合检索
# ================================================================

class _FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return list(self.scores)


def _chunk(source, content, page=1):
    return {"content": content, "source": source, "page": page, "type": "text"}


@pytest.fixture
def multi_env(monkeypatch):
    """可按查询区分向量结果的假环境，并捕获送进精排的候选池与精排调用次数。"""
    from src import reranker

    box = {
        "vector_by_query": {},   # {query: [chunk, ...]}（顺序即排名）
        "chunks": [],
        "bm25_scores": None,     # None = BM25 路不可用
        "rerank_calls": [],      # [(query, candidates, top_k), ...]
    }

    def fake_search_similar(kb_id, query, top_k=5):
        results = [dict(c) for c in box["vector_by_query"].get(query, [])][:top_k]
        for rank, r in enumerate(results, start=1):
            r["vector_rank"] = rank
        return results

    def fake_get_bm25(kb_id):
        if box["bm25_scores"] is None:
            return None, []
        return _FakeBM25(box["bm25_scores"]), box["chunks"]

    def spy_rerank(query, candidates, top_k, enabled=None):
        box["rerank_calls"].append((query, list(candidates), top_k))
        return candidates[:top_k]

    monkeypatch.setattr(vector_store, "search_similar", fake_search_similar)
    monkeypatch.setattr(reranker, "rerank", spy_rerank)
    monkeypatch.setattr(hr, "_get_bm25", fake_get_bm25)
    monkeypatch.setattr(hr, "BM25_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "VECTOR_WEIGHT", 1.0)
    monkeypatch.setattr(hr, "RETRIEVE_CANDIDATES", 5)
    return box


_KB = "kb-test"
_A = _chunk("doc.md", "AAAAAAAA" * 10)   # 两个查询都命中（共识块）
_B = _chunk("doc.md", "BBBBBBBB" * 10)   # 只有 q1 命中
_C = _chunk("doc.md", "CCCCCCCC" * 10)   # 只有 q2 命中


def test_search_multi_consensus_beats_single_hit(multi_env):
    """跨查询共识：被两个表述都命中的块，融合分要高于只有单路命中的块。"""
    multi_env["vector_by_query"] = {"q1": [_A, _B], "q2": [_A, _C]}

    retriever = HybridRetriever(_KB)
    out = retriever.search_multi(["q1", "q2"], top_k=2)

    assert [r["content"] for r in out][0] == _A["content"]


def test_search_multi_rerank_once_with_primary_query(multi_env):
    """精排只调一次（不是每个查询各一次），且打分用主查询 queries[0]。"""
    multi_env["vector_by_query"] = {"q1": [_A], "q2": [_B], "q3": [_C]}

    retriever = HybridRetriever(_KB)
    retriever.search_multi(["q1", "q2", "q3"], top_k=3)

    assert len(multi_env["rerank_calls"]) == 1
    assert multi_env["rerank_calls"][0][0] == "q1"


def test_search_multi_dedups_across_queries(multi_env):
    """同一块被多个查询召回：融合结果里只出现一次，RRF 分数跨查询累加。"""
    multi_env["vector_by_query"] = {"q1": [_A, _B], "q2": [_A]}

    retriever = HybridRetriever(_KB)
    retriever.search_multi(["q1", "q2"], top_k=5)

    _, pool, _ = multi_env["rerank_calls"][0]
    contents = [r["content"] for r in pool]
    assert contents.count(_A["content"]) == 1

    a_score = next(r["rrf_score"] for r in pool if r["content"] == _A["content"])
    b_score = next(r["rrf_score"] for r in pool if r["content"] == _B["content"])
    # A 在两路里都是 rank1：1/61 + 1/61；B 只在 q1 里 rank2：1/62
    assert a_score == pytest.approx(2 / 61)
    assert b_score == pytest.approx(1 / 62)
    assert a_score > b_score


def test_search_multi_single_query_matches_search(multi_env):
    """单元素列表的行为与 search() 完全一致（降级路径的等价性）。"""
    multi_env["vector_by_query"] = {"q1": [_A, _B]}

    retriever = HybridRetriever(_KB)
    via_multi = retriever.search_multi(["q1"], top_k=2)
    via_search = retriever.search("q1", top_k=2)

    assert [(r["content"], r["rrf_score"]) for r in via_multi] == \
           [(r["content"], r["rrf_score"]) for r in via_search]


def test_search_multi_truncates_merged_pool(multi_env):
    """N 路各出 candidate_k 条，合并后必须截断回 candidate_k（精排预算不随查询数涨）。"""
    lots = [_chunk("doc.md", f"块{i}" + "x" * 40) for i in range(8)]
    # q1 / q2 各召回 6 个不同块（candidate_k=5，各截到 5）
    multi_env["vector_by_query"] = {"q1": lots[:6], "q2": lots[2:8]}

    retriever = HybridRetriever(_KB)
    retriever.search_multi(["q1", "q2"], top_k=3)

    _, pool, _ = multi_env["rerank_calls"][0]
    assert len(pool) <= 5  # 截断回候选池大小，不是 10
