"""CRAG 生成后自检（src/answer_verifier.py + rag_qa 接入）单测。

守的行为（对应 docs/technical-optimization-plan.md §5.5，另加接入层行为）：
  · verify_answer：关闭 / 无上下文 / LLM 失败 → 返回 5（失败即放行，绝不阻断回答）；
    正常打分；分数截断到 1-5；容忍尾随句点
  · rag_query 接入：低于阈值触发严格重试、取分高者；重试用 STRICT 提示词
  · 流式接入：低于阈值的答案不写缓存（已输出无法撤回，但缓存会把问题放大成常态）

全部 monkeypatch，不调真实 LLM（CI 约定）。
⚠️ src.rag_qa 不能在模块顶层 import（langfuse 缺失会在 pytest collection 阶段
炸掉整个文件），必须在 stub_langfuse 夹具之后导入并经 env 字典传递——
test_direct_return.py 踩过同一个坑。
"""
import pytest

import src.answer_verifier as verifier
import src.query_cache as qc  # rag_qa 函数内惰性导入 → patch 必须打到源模块


# ================================================================
# verify_answer：打分器
# ================================================================

def _make_client(reply):
    """假 OpenAI 客户端：reply 为 str 时照常返回，为 Exception 时抛出。"""
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


def test_disabled_returns_5_without_llm(monkeypatch):
    """开关关闭：返回满分 5，且根本不该构造/调用 LLM 客户端。"""
    monkeypatch.setattr(verifier, "ANSWER_VERIFY_ENABLED", False)

    def _boom():
        raise AssertionError("关闭时不应构造/调用 LLM 客户端")

    monkeypatch.setattr(verifier, "_get_client", _boom)
    assert verifier.verify_answer("答案", ["上下文"]) == 5


def test_empty_contexts_returns_5(monkeypatch):
    """没有上下文（门控已清空）：无从验证，直接放行。"""
    monkeypatch.setattr(verifier, "ANSWER_VERIFY_ENABLED", True)

    def _boom():
        raise AssertionError("无上下文时不应调用 LLM")

    monkeypatch.setattr(verifier, "_get_client", _boom)
    assert verifier.verify_answer("答案", []) == 5


def test_score_four(monkeypatch):
    """mock LLM 返回 "4" → 4（对应文档测试用例 2）。"""
    monkeypatch.setattr(verifier, "ANSWER_VERIFY_ENABLED", True)
    client, _ = _make_client("4")
    monkeypatch.setattr(verifier, "_get_client", lambda: client)
    assert verifier.verify_answer("答案", ["上下文"]) == 4


def test_score_two(monkeypatch):
    """mock LLM 返回 "2" → 2（对应文档测试用例 3）。"""
    monkeypatch.setattr(verifier, "ANSWER_VERIFY_ENABLED", True)
    client, _ = _make_client("2")
    monkeypatch.setattr(verifier, "_get_client", lambda: client)
    assert verifier.verify_answer("答案", ["上下文"]) == 2


def test_llm_exception_fails_open(monkeypatch):
    """LLM 抛异常 → 返回 5 放行（对应文档测试用例 4）：验证挂了不能阻断回答。"""
    monkeypatch.setattr(verifier, "ANSWER_VERIFY_ENABLED", True)
    client, _ = _make_client(RuntimeError("API 限流"))
    monkeypatch.setattr(verifier, "_get_client", lambda: client)
    assert verifier.verify_answer("答案", ["上下文"]) == 5


def test_score_clamped_and_dot_tolerated(monkeypatch):
    """越界分数截断到 [1,5]；"4." 这类尾随句点也能解析。"""
    monkeypatch.setattr(verifier, "ANSWER_VERIFY_ENABLED", True)

    for reply, expected in [("9", 5), ("0", 1), ("4.", 4)]:
        client, _ = _make_client(reply)
        monkeypatch.setattr(verifier, "_get_client", lambda c=client: c)
        assert verifier.verify_answer("答案", ["上下文"]) == expected, reply


# ================================================================
# rag_qa 接入：严格重试 / 流式缓存把关
# ================================================================

@pytest.fixture
def env(monkeypatch, stub_langfuse):
    """搭一条最小问答链路：检索替身 + 生成替身 + 验证替身 + 缓存/日志 spies。"""
    import src.rag_qa as rag_qa  # 必须在 stub_langfuse 之后 import

    monkeypatch.setattr(rag_qa, "ANSWER_VERIFY_ENABLED", True)
    monkeypatch.setattr(rag_qa, "ANSWER_VERIFY_THRESHOLD", 3)

    hits = [{"content": "上下文A", "source": "a.md", "rerank_score": 0.8}]
    retriever = type("R", (), {"search": lambda self, q, top_k=5, rerank=None: [dict(h) for h in hits]})()
    monkeypatch.setattr(rag_qa, "HybridRetriever", lambda kb_id: retriever)

    generated = []
    verify_scores = []

    def fake_generate(query, contexts, **kw):
        generated.append({"kw": kw, "contexts": contexts})
        return "答案v1"

    monkeypatch.setattr(rag_qa, "generate_answer", fake_generate)
    monkeypatch.setattr(rag_qa, "verify_answer",
                        lambda answer, contexts: verify_scores.pop(0))
    monkeypatch.setattr(rag_qa, "_log_answer", lambda **kw: 1)
    monkeypatch.setattr(qc, "get_cached_answer", lambda *a, **k: None)
    cached = []
    monkeypatch.setattr(qc, "cache_answer", lambda *a, **k: cached.append((a, k)))

    return {"rag_qa": rag_qa, "hits": hits, "generated": generated,
            "verify_scores": verify_scores, "cached": cached}


def test_rag_query_retry_uses_strict_prompt(env, monkeypatch):
    """初版 2 分 < 阈值 3 → 触发严格重试；重试版 4 分更高 → 采用重试版。"""
    rag_qa = env["rag_qa"]
    env["verify_scores"].extend([2, 4])  # 初版 2 分，重试版 4 分

    def fake_stream(*a, **kw):
        raise AssertionError("rag_query 不应走流式生成")

    monkeypatch.setattr(rag_qa, "stream_generate_answer", fake_stream)
    result = rag_qa.rag_query("kb", "问题")

    # 两次生成：初版（无 override）+ 严格重试（override=STRICT 提示词）
    assert len(env["generated"]) == 2
    assert env["generated"][0]["kw"].get("system_prompt_override") is None
    assert env["generated"][1]["kw"].get("system_prompt_override") == \
        rag_qa.STRICT_ANSWER_SYSTEM_PROMPT
    assert result["answer"] == "答案v1"


def test_rag_query_keeps_original_when_retry_not_better(env, monkeypatch):
    """重试版分数不比初版高 → 保留初版（重试不是盲目替换）。"""
    rag_qa = env["rag_qa"]
    env["verify_scores"].extend([2, 2])  # 初版 2，重试版也不行

    result = rag_qa.rag_query("kb", "问题")
    assert len(env["generated"]) == 2  # 重试发生了
    assert result["answer"] == "答案v1"


def test_rag_query_no_verify_when_disabled(env, monkeypatch):
    """开关关闭（生产默认）：一次生成就返回，零额外调用。"""
    rag_qa = env["rag_qa"]
    monkeypatch.setattr(rag_qa, "ANSWER_VERIFY_ENABLED", False)
    env["verify_scores"].append(0)  # 若 verify 被调用会拿到 0 并暴露

    result = rag_qa.rag_query("kb", "问题")
    assert len(env["generated"]) == 1
    assert result["answer"] == "答案v1"


def test_stream_skips_cache_when_unverified(env, monkeypatch):
    """流式路径：答案低于阈值 → 不写缓存（答案已输出无法撤回，但缓存能拦）。"""
    rag_qa = env["rag_qa"]
    env["verify_scores"].extend([2])  # 流式答案 2 分

    def fake_stream_llm(query, contexts, **kw):
        yield "不靠谱的"
        yield "流式答案"

    monkeypatch.setattr(rag_qa, "stream_generate_answer", fake_stream_llm)
    chunks = list(rag_qa.stream_rag_query("kb", "问题")[0])

    assert "".join(chunks) == "不靠谱的流式答案"
    assert env["cached"] == []  # 不达标 → 不写缓存


def test_stream_caches_when_verified(env, monkeypatch):
    """流式路径：验证达标 → 正常写缓存（自检不能把正常路径也搞坏）。"""
    rag_qa = env["rag_qa"]
    env["verify_scores"].extend([4])

    def fake_stream_llm(query, contexts, **kw):
        yield "靠谱的"
        yield "流式答案"

    monkeypatch.setattr(rag_qa, "stream_generate_answer", fake_stream_llm)
    list(rag_qa.stream_rag_query("kb", "问题")[0])

    assert len(env["cached"]) == 1
