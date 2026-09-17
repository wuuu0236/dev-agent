"""Adaptive Routing（src/query_router.py + rag_qa 接入）单测。

守的行为（对应 docs/technical-optimization-plan.md §7.5，另加接入层与回归）：
  · classify_query：关闭 → simple；规则引擎零成本拦明显闲聊；
    LLM 分类兜底；无效输出 / 异常一律降级 simple（路由挂了不能拖垮问答）
  · 天气/时间类关键词必须锚定句首——「上传时间是什么时候」不是闲聊（误杀回归）
  · decompose_query：正常拆分 / 异常退回 [query]
  · rag_query 接入：chitchat 不检索且走 NO_CONTEXT 生成（拒答 ≠ 拒绝服务）；
    multi-hop 走 search_multi（原问题打头，精排一次）
  · stream 接入：chitchat 分支必须赋值 gate_score（回归：漏赋值是 NameError）

全部 monkeypatch，不调真实 LLM（CI 约定）。
⚠️ src.rag_qa 不能在模块顶层 import（langfuse 缺失会炸 collection），
必须在 stub_langfuse 夹具之后导入——test_direct_return.py 同款坑。
"""
import pytest

import src.query_router as qr
import src.query_cache as qc  # rag_qa 函数内惰性导入 → patch 必须打到源模块


# ================================================================
# classify_query：分类
# ================================================================

def test_disabled_returns_simple_without_llm(monkeypatch):
    """开关关闭：恒 simple，零成本，不该碰 LLM。"""
    monkeypatch.setattr(qr, "ADAPTIVE_ROUTING_ENABLED", False)

    def _boom():
        raise AssertionError("关闭时不应构造/调用 LLM 客户端")

    monkeypatch.setattr(qr, "_get_client", _boom)
    assert qr.classify_query("你好") == "simple"


def _enable_with_reply(monkeypatch, reply):
    """打开路由开关并塞入固定回复的假客户端，返回调用记录。"""
    monkeypatch.setattr(qr, "ADAPTIVE_ROUTING_ENABLED", True)
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

    monkeypatch.setattr(qr, "_get_client", lambda: _FakeClient())
    return calls


def test_greeting_hits_rules_without_llm(monkeypatch):
    """「你好」被规则引擎拦下 → chitchat，零 LLM 调用（对应文档测试用例 2）。"""
    calls = _enable_with_reply(monkeypatch, "simple")

    def _boom():
        raise AssertionError("规则命中后不应再调 LLM")

    monkeypatch.setattr(qr, "_get_client", _boom)
    assert qr.classify_query("你好") == "chitchat"
    assert calls == []


def test_time_question_not_chitchat(monkeypatch):
    """回归：「上传时间是什么时候」是正经知识库问题，不是闲聊——
    天气/时间类关键词必须锚定句首，不锚定会把它误杀成 chitchat。"""
    _enable_with_reply(monkeypatch, "simple")
    assert qr.classify_query("上传时间是什么时候") == "simple"


def test_simple_question_via_llm(monkeypatch):
    """普通问题交给 LLM 分类 → simple（对应文档测试用例 3）。"""
    calls = _enable_with_reply(monkeypatch, "simple")
    assert qr.classify_query("混合检索是怎么做的") == "simple"
    assert calls[0]["model"] == qr.LLM_MODEL


def test_multi_hop_via_llm(monkeypatch):
    """LLM 判定多跳 → multi-hop（对应文档测试用例 4）。"""
    _enable_with_reply(monkeypatch, "multi-hop")
    assert qr.classify_query("A 的作者和B 的作者是什么关系") == "multi-hop"


def test_invalid_llm_output_falls_back_to_simple(monkeypatch):
    """LLM 输出不在三个合法值里 → simple（分类降级，不放行奇怪的值）。"""
    _enable_with_reply(monkeypatch, "我觉得这是个好问题")
    assert qr.classify_query("任意问题") == "simple"


def test_llm_exception_falls_back_to_simple(monkeypatch):
    """LLM 抛异常 → simple：路由挂了问答照常走。"""
    _enable_with_reply(monkeypatch, RuntimeError("API 限流"))
    assert qr.classify_query("任意问题") == "simple"


# ================================================================
# decompose_query：多跳分解
# ================================================================

def test_decompose_returns_lines(monkeypatch):
    """mock 返回 2 行 → 2 个子问题（对应文档测试用例 5）。"""
    _enable_with_reply(monkeypatch, "谁是 A 的作者\n谁是 B 的作者")
    out = qr.decompose_query("A 的作者和B 的作者是什么关系")
    assert out == ["谁是 A 的作者", "谁是 B 的作者"]


def test_decompose_exception_returns_original(monkeypatch):
    """分解异常 → [query]（对应文档测试用例 6）。"""
    _enable_with_reply(monkeypatch, RuntimeError("超时"))
    assert qr.decompose_query("原问题") == ["原问题"]


# ================================================================
# rag_qa 接入：chitchat 分流 / multi-hop 联合检索
# ================================================================

@pytest.fixture
def env(monkeypatch, stub_langfuse):
    """最小问答链路：可编程的假检索器 + 生成替身 + 日志/缓存 spies。"""
    import src.rag_qa as rag_qa  # 必须在 stub_langfuse 之后 import

    hits = [{"content": "上下文A", "source": "a.md", "rerank_score": 0.8}]
    calls = {"search": [], "search_multi": []}

    class _FakeRetriever:
        def search(self, query, top_k=5, rerank=None):
            calls["search"].append(query)
            return [dict(h) for h in hits]

        def search_multi(self, queries, top_k=5, rerank=None):
            calls["search_multi"].append(list(queries))
            return [dict(h) for h in hits]

    monkeypatch.setattr(rag_qa, "HybridRetriever", lambda kb_id: _FakeRetriever())

    generated = []
    monkeypatch.setattr(rag_qa, "generate_answer",
                        lambda query, contexts, **kw: generated.append({"kw": kw}) or "答案v1")

    def fake_stream(query, contexts, **kw):
        generated.append({"kw": kw, "stream": True})
        yield "答案v1"

    monkeypatch.setattr(rag_qa, "stream_generate_answer", fake_stream)

    logs = []
    monkeypatch.setattr(rag_qa, "_log_answer", lambda **kw: logs.append(kw) or 1)
    monkeypatch.setattr(qc, "get_cached_answer", lambda *a, **k: None)
    monkeypatch.setattr(qc, "cache_answer", lambda *a, **k: None)

    return {"rag_qa": rag_qa, "hits": hits, "calls": calls,
            "generated": generated, "logs": logs}


def test_rag_query_chitchat_skips_retrieval(env, monkeypatch):
    """chitchat：一次检索都不发生，走 NO_CONTEXT 生成，日志 gate_score=None。"""
    rag_qa = env["rag_qa"]
    monkeypatch.setattr(rag_qa, "classify_query", lambda q: "chitchat")

    result = rag_qa.rag_query("kb", "你好")

    assert env["calls"]["search"] == [] and env["calls"]["search_multi"] == []
    assert result["grounded"] is False and result["contexts"] == []
    assert env["generated"][0]["kw"]["grounded"] is False
    assert env["logs"][0]["gate_score"] is None  # "没检索"，不冒充 0.0


def test_rag_query_multi_hop_uses_search_multi_with_original_first(env, monkeypatch):
    """multi-hop：search_multi 收到 [原问题, 子问题...]（原问题打头做精排主查询）。"""
    rag_qa = env["rag_qa"]
    monkeypatch.setattr(rag_qa, "classify_query", lambda q: "multi-hop")
    monkeypatch.setattr(rag_qa, "decompose_query", lambda q: ["子问题一", "子问题二"])

    rag_qa.rag_query("kb", "原问题")

    assert env["calls"]["search_multi"] == [["原问题", "子问题一", "子问题二"]]


def test_rag_query_decompose_failure_falls_back_to_single(env, monkeypatch):
    """分解失败（返回 [query]）→ 退回普通单查询路径。"""
    rag_qa = env["rag_qa"]
    monkeypatch.setattr(rag_qa, "classify_query", lambda q: "multi-hop")
    monkeypatch.setattr(rag_qa, "decompose_query", lambda q: ["原问题"])

    rag_qa.rag_query("kb", "原问题")

    assert env["calls"]["search"] == ["原问题"]
    assert env["calls"]["search_multi"] == []


def test_rag_query_disabled_routes_to_normal_search(env, monkeypatch):
    """路由关闭（生产默认）：classify_query 返回 simple，正常单查询检索。"""
    rag_qa = env["rag_qa"]
    monkeypatch.setattr(rag_qa, "classify_query", lambda q: "simple")

    result = rag_qa.rag_query("kb", "问题")

    assert env["calls"]["search"] == ["问题"]
    assert result["grounded"] is True


def test_stream_chitchat_does_not_crash_and_logs_none(env, monkeypatch):
    """回归：stream 的 chitchat 分支漏赋 gate_score 是 NameError——
    本测试钉死「整条流式链路跑得完 + 日志 gate_score=None」。"""
    rag_qa = env["rag_qa"]
    monkeypatch.setattr(rag_qa, "classify_query", lambda q: "chitchat")

    gen, sources, contexts, retrieval_query, log_ref = rag_qa.stream_rag_query("kb", "你好")
    chunks = list(gen)

    assert "".join(chunks) == "答案v1"
    assert env["logs"][0]["gate_score"] is None
    assert env["calls"]["search"] == [] and env["calls"]["search_multi"] == []
