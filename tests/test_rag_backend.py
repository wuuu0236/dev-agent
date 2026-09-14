"""推理后端分派（云端 DeepSeek / 本地 Ollama）的回归测试。

存在理由：`_get_ollama_client` 里把 `OpenAI` 写成了 `OpenAIClient`（这个类在本文件
里根本不存在）。后果是**只要选 ollama 后端，第一次调用必然 NameError**——整条
「本地私有化 / 离线」路径一用就崩。而云端路径完全正常，所以线上 Demo 一直没露馅，
README 宣传的离线能力却是死的。

这类错字在静态检查缺位时能活很久（本机没有 ruff / pyflakes 也会漏掉），
所以留一条能跑的行为测试，而不是只靠注释提醒。

主链路要 import rag_qa（依赖 langfuse），用 conftest 的 stub_langfuse 夹具补缺。
"""
import pytest


@pytest.fixture
def rag(monkeypatch, stub_langfuse):
    import src.rag_qa as rag_qa
    # 隔离掉进程级客户端缓存，避免测试互相污染
    monkeypatch.setattr(rag_qa, "_ollama_clients", {})
    monkeypatch.setattr(rag_qa, "_client", None)
    return rag_qa


class _FakeOpenAI:
    """替身 OpenAI 客户端，记下构造参数。"""

    def __init__(self, **kw):
        self.kw = kw


def test_ollama_backend_does_not_reference_a_nonexistent_class(rag, monkeypatch):
    """回归：`_get_ollama_client` 必须能用真名 OpenAI 构造出客户端（原来会 NameError）。"""
    monkeypatch.setattr(rag, "OpenAI", _FakeOpenAI)

    client = rag._get_ollama_client("http://localhost:11434/v1")

    assert isinstance(client, _FakeOpenAI)
    assert client.kw["base_url"] == "http://localhost:11434/v1"
    assert client.kw["api_key"] == "ollama", "本地 Ollama 不校验 key，但字段不能缺"


def test_ollama_client_is_cached_per_base_url(rag, monkeypatch):
    """按 base_url 缓存客户端：同一地址不重复构造，不同地址互不串用。"""
    monkeypatch.setattr(rag, "OpenAI", _FakeOpenAI)

    a1 = rag._get_ollama_client("http://localhost:11434/v1")
    a2 = rag._get_ollama_client("http://localhost:11434/v1")
    b = rag._get_ollama_client("http://192.168.1.9:11434/v1")

    assert a1 is a2
    assert b is not a1


def test_backend_picks_matching_model_name(rag, monkeypatch):
    """选后端必须同时选中该后端的模型名——配错了会拿着 ollama 的模型名去请求云端。"""
    from src.config import LLM_MODEL, OLLAMA_LLM_MODEL
    monkeypatch.setattr(rag, "OpenAI", _FakeOpenAI)

    _, model_cloud, _ = rag._prepare_generation("Q", [], backend="cloud", grounded=False)
    _, model_local, _ = rag._prepare_generation("Q", [], backend="ollama", grounded=False)

    assert model_cloud == LLM_MODEL
    assert model_local == OLLAMA_LLM_MODEL
    assert model_cloud != model_local, "两个后端的模型名撞车了，需要检查 config"


def test_cloud_backend_uses_cloud_client(rag, monkeypatch):
    """cloud 后端不该走到 Ollama 客户端（否则会把 DeepSeek 的请求发到 localhost）。"""
    sentinel = _FakeOpenAI(api_key="cloud")
    monkeypatch.setattr(rag, "_get_client", lambda: sentinel)
    monkeypatch.setattr(rag, "OpenAI", _FakeOpenAI)

    client, _, _ = rag._prepare_generation("Q", [], backend="cloud", grounded=False)

    assert client is sentinel


def test_ungrounded_injects_no_documents(rag):
    """门控未通过时一份参考文档都不注入（低分上下文只会诱发幻觉）。"""
    contexts = [{"source": "a.md", "page": 1, "content": "看起来相关但不够",
                 "rerank_score": 0.02}]

    _, _, messages = rag._prepare_generation("问题", contexts, backend="cloud",
                                             grounded=False)

    joined = "\n".join(m["content"] for m in messages)
    assert "a.md" not in joined, "未过门控却把低分文档注入进去了"
    assert "看起来相关但不够" not in joined
    assert messages[-1]["content"].strip() == "问题"
