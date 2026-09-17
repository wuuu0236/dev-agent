"""高相似直接返回（docs/technical-optimization-plan.md 第六章）行为测试。

守三条：
  · 触发时**不调 LLM**——省一次调用是这个功能存在的意义，两边路径（流式/非流式）都一样
  · 返回值结构与常规路径完全一致（rag_query 7 键 / stream 5 元组）——文档全局红线
  · 默认关闭时行为与改动前零差异；分数低于阈值时也零差异

主链路假件模式沿用 tests/test_answer_log.py：检索换成假类，LLM 与日志是记录器，
缓存双函数封死（否则流式 direct 分支会把测试写进真实缓存表）。
"""
import pytest

import src.query_cache as qc

# rag_qa 顶层 import langfuse —— 不能在模块顶层 import（CI / 本机都可能没装），
# 必须等 stub_langfuse 夹具补完缺再 import（沿用 tests/test_answer_log.py 的做法）。

# 精排 0.95：远超 DIRECT_RETURN_THRESHOLD(0.92)，也必然过门控(0.3)
_HIGH = [
    {"source": "手册.md", "page": 3, "type": "text",
     "content": "BM25 与向量两路召回互补，RRF 融合后截断回候选池。",
     "rerank_score": 0.95, "rrf_score": 0.018},
]
# 精排 0.5：过门控但够不着直接返回阈值 —— 必须照常走 LLM
_LOW = [
    {"source": "手册.md", "page": 1, "type": "text",
     "content": "相关但没高到可以照搬的内容", "rerank_score": 0.5, "rrf_score": 0.016},
]


@pytest.fixture
def env(monkeypatch, stub_langfuse):
    """rag_qa 主链路假件：检索返回可配置 hits，LLM / 日志都是记录器。"""
    import src.rag_qa as rag_qa

    hits = {"list": [dict(c) for c in _HIGH]}

    class FakeRetriever:
        def __init__(self, kb_id):
            pass

        def search(self, query, top_k=None):
            return [dict(c) for c in hits["list"]]

    llm_calls = []

    def fake_llm(*a, **kw):
        llm_calls.append({"args": a, "kw": kw})
        return "LLM 生成的答案"

    def fake_stream_llm(*a, **kw):
        llm_calls.append({"args": a, "kw": kw, "stream": True})
        return iter(["LLM", "答案"])

    log_calls = []
    monkeypatch.setattr(rag_qa, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(rag_qa, "generate_answer", fake_llm)
    # 记录进同一个 llm_calls：流式 / 非流式任一被调用都算"走了 LLM"
    monkeypatch.setattr(rag_qa, "stream_generate_answer", fake_stream_llm)
    monkeypatch.setattr(rag_qa, "_log_answer", lambda **kw: log_calls.append(kw) or 42)
    monkeypatch.setattr(qc, "get_cached_answer", lambda *a, **k: None)
    monkeypatch.setattr(qc, "cache_answer", lambda *a, **k: None)
    return {"rag_qa": rag_qa, "hits": hits, "llm_calls": llm_calls, "log_calls": log_calls}


class TestUnit:
    """判断函数本身的分界：开关 / 阈值 / 空结果。"""

    def test_disabled_returns_none_even_at_high_score(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", False)
        assert rag_qa._direct_return_answer([dict(c) for c in _HIGH]) is None

    def test_high_score_returns_verbatim_with_citation(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", True)
        out = rag_qa._direct_return_answer([dict(c) for c in _HIGH])
        assert out is not None
        assert "RRF 融合后截断回候选池" in out          # 原文照搬
        assert "[1]" in out                              # 引用序号留给前端联动
        assert "手册.md" in out and "第3页" in out       # 位置信息帮用户定位

    def test_low_score_or_empty_returns_none(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", True)
        assert rag_qa._direct_return_answer([dict(c) for c in _LOW]) is None
        assert rag_qa._direct_return_answer([]) is None


class TestRagQueryPath:
    def test_direct_return_skips_llm_and_keeps_contract(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", True)
        result = rag_qa.rag_query("kb1", "混合检索为什么两路都开？")

        assert env["llm_calls"] == [], "触发直接返回时不得调 LLM"
        assert "RRF 融合后截断回候选池" in result["answer"]
        # 返回值结构红线：7 个键一个不少；引用与门控语义照常
        assert set(result) == {"answer", "sources", "contexts", "grounded",
                               "query", "retrieval_query", "log_id"}
        assert result["grounded"] is True and result["contexts"]
        assert result["sources"] and result["sources"][0]["source"] == "手册.md"
        assert result["log_id"] == 42
        assert env["log_calls"] and env["log_calls"][0]["backend"] == "direct_return"

    def test_below_threshold_still_uses_llm(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", True)
        env["hits"]["list"] = [dict(c) for c in _LOW]
        result = rag_qa.rag_query("kb1", "问题")
        assert env["llm_calls"], "分数低于阈值必须照常走 LLM"
        assert result["answer"] == "LLM 生成的答案"

    def test_disabled_by_default_behaves_as_before(self, env):
        """默认关闭（config 默认 false）= 行为与改动前完全一致。"""
        rag_qa = env["rag_qa"]
        result = rag_qa.rag_query("kb1", "问题")
        assert env["llm_calls"] and result["answer"] == "LLM 生成的答案"
        assert env["log_calls"][0]["backend"] != "direct_return"


class TestStreamPath:
    def test_stream_direct_yields_answer_without_llm(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", True)
        gen, sources, contexts, retrieval_query, log_ref = rag_qa.stream_rag_query("kb1", "问题")

        out = "".join(gen)
        assert "RRF 融合后截断回候选池" in out
        assert env["llm_calls"] == [], "流式路径同样不得调 LLM"
        # 5 元组契约不变；log_ref 与 gate_score 照常回填
        assert log_ref["id"] == 42
        assert log_ref["gate_score"] == pytest.approx(0.95)
        assert sources and sources[0]["source"] == "手册.md"

    def test_stream_below_threshold_uses_llm(self, env, monkeypatch):
        rag_qa = env["rag_qa"]
        monkeypatch.setattr(rag_qa, "DIRECT_RETURN_ENABLED", True)
        env["hits"]["list"] = [dict(c) for c in _LOW]
        gen, *_ = rag_qa.stream_rag_query("kb1", "问题")
        assert "".join(gen) == "LLM答案"
        assert env["llm_calls"]
