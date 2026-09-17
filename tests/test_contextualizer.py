"""Contextual Retrieval（src/contextualizer.py）单测。

守的四条行为（对应 docs/technical-optimization-plan.md §2.6）：
  1. 开关关闭 → generate_context 返回空串、且根本不该碰 LLM 客户端
  2. LLM 正常返回描述 → chunk content 前面出现 `[描述] ` 前缀
  3. LLM 回答"自成一体" → 不加前缀（自包含片段不该被硬加语境）
  4. LLM 抛异常 → 该块不加前缀、整批不中断（单块失败只影响自己）

约定：所有 LLM 调用走替身（fake client），测试不依赖任何真实凭据（CI 约定）。
"""
import pytest

import src.contextualizer as contextualizer


def _make_client(reply):
    """构造一个假 OpenAI 客户端：reply 为 str 时照常返回，为 Exception 时抛出。"""
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

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    return _FakeClient(), calls


@pytest.fixture
def enabled(monkeypatch):
    """打开开关 + 塞进假客户端，返回调用记录列表供断言。"""
    monkeypatch.setattr(contextualizer, "CONTEXTUAL_RETRIEVAL_ENABLED", True)
    calls: list[dict] = []
    return calls


def test_disabled_returns_empty_without_llm_call(monkeypatch):
    """开关关闭：直接返回空串，且一个 LLM 客户端都不该构造。"""
    monkeypatch.setattr(contextualizer, "CONTEXTUAL_RETRIEVAL_ENABLED", False)

    def _boom():
        raise AssertionError("开关关闭时不应构造/调用 LLM 客户端")

    monkeypatch.setattr(contextualizer, "_get_client", _boom)
    assert contextualizer.generate_context("某段内容", "整篇文档") == ""


def test_disabled_contextualize_is_noop(monkeypatch):
    """开关关闭：contextualize_chunks 原样返回，chunk 内容一个字都不动。"""
    monkeypatch.setattr(contextualizer, "CONTEXTUAL_RETRIEVAL_ENABLED", False)
    chunks = [{"content": "原文", "chunk_index": 0}]
    out = contextualizer.contextualize_chunks(chunks, "整篇文档")
    assert out[0]["content"] == "原文"


def test_normal_description_prepends_prefix(enabled, monkeypatch):
    """LLM 正常返回描述：content 变成 `[描述] 原文`，原文完整保留在前缀之后。"""
    client, calls = _make_client("这一段介绍 BM25 混合检索的实现细节。")
    monkeypatch.setattr(contextualizer, "_get_client", lambda: client)

    context = contextualizer.generate_context("chunk 正文", "整篇文档")
    assert context == "这一段介绍 BM25 混合检索的实现细节。"

    chunks = [{"content": "chunk 正文", "chunk_index": 0}]
    out = contextualizer.contextualize_chunks(chunks, "整篇文档")
    assert out[0]["content"] == "[这一段介绍 BM25 混合检索的实现细节。] chunk 正文"

    # 送审结构：system + user 两条消息，user 里同时带文档与片段
    # （generate_context + contextualize_chunks 各调一次，检查第一次的送审结构）
    assert len(calls) == 2
    messages = calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "<document>" in messages[1]["content"]
    assert "<chunk>" in messages[1]["content"]
    assert calls[0]["model"] == contextualizer.CONTEXTUAL_MODEL


def test_document_truncated_to_max_chars(enabled, monkeypatch):
    """整篇文档超过 CONTEXTUAL_DOCUMENT_MAX_CHARS 时只送前 N 个字符。"""
    monkeypatch.setattr(contextualizer, "CONTEXTUAL_DOCUMENT_MAX_CHARS", 10)
    client, calls = _make_client("描述")
    monkeypatch.setattr(contextualizer, "_get_client", lambda: client)

    contextualizer.generate_context("chunk", "甲" * 500)
    user_content = calls[0]["messages"][1]["content"]
    assert "甲" * 10 in user_content
    assert "甲" * 11 not in user_content


def test_self_contained_reply_adds_no_prefix(enabled, monkeypatch):
    """模型自认"自成一体"：返回空串，chunk 内容保持原样。"""
    client, _ = _make_client("此片段自成一体，无需额外上下文。")
    monkeypatch.setattr(contextualizer, "_get_client", lambda: client)

    assert contextualizer.generate_context("自包含的完整段落", "整篇文档") == ""
    chunks = [{"content": "自包含的完整段落"}]
    out = contextualizer.contextualize_chunks(chunks, "整篇文档")
    assert out[0]["content"] == "自包含的完整段落"


def test_llm_exception_does_not_break_batch(enabled, monkeypatch):
    """LLM 抛异常：该块不加前缀、不向上抛——单块失败只影响自己，整批照常。"""
    client, _ = _make_client(RuntimeError("API 限流"))
    monkeypatch.setattr(contextualizer, "_get_client", lambda: client)

    assert contextualizer.generate_context("任意", "整篇文档") == ""
    chunks = [{"content": "块一"}, {"content": "块二"}]
    out = contextualizer.contextualize_chunks(chunks, "整篇文档")
    assert [c["content"] for c in out] == ["块一", "块二"]


def test_mixed_batch_partial_success(enabled, monkeypatch):
    """一批里第一个块成功、第二个块失败：各自独立，互不拖累。"""
    replies = iter(["第一块的语境描述"])
    failures = iter([RuntimeError("第二个块运气不好")])

    class _FlakyCompletions:
        def create(self, **kwargs):
            try:
                reply = next(replies)
            except StopIteration:
                raise next(failures)

            class _Msg:
                content = reply

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _FlakyClient:
        class chat:
            completions = _FlakyCompletions()

    monkeypatch.setattr(contextualizer, "_get_client", lambda: _FlakyClient())

    chunks = [{"content": "块一", "chunk_index": 0}, {"content": "块二", "chunk_index": 1}]
    out = contextualizer.contextualize_chunks(chunks, "整篇文档")
    assert out[0]["content"] == "[第一块的语境描述] 块一"
    assert out[1]["content"] == "块二"


def test_build_full_document_joins_parsed_texts():
    """build_full_document：把 parse_file 输出各段的 text 拼成整篇文档。"""
    parsed = [
        {"text": "第一段内容", "page": 1, "source": "a.pdf"},
        {"text": "第二段内容", "page": 2, "source": "a.pdf"},
        {"text": "", "page": 3, "source": "a.pdf"},   # 空段被跳过
    ]
    assert contextualizer.build_full_document(parsed) == "第一段内容\n\n第二段内容"
    assert contextualizer.build_full_document([]) == ""
