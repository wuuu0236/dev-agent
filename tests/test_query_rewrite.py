"""query_rewrite（追问消解）单元测试。

全部用假 client，不调真实 LLM。覆盖三条主分支：
  · 不该改写（无历史 / 开关关 / 历史里没有用户消息）→ 原样返回且不调模型
  · 该改写 → 返回模型输出，且提示词里带上了历史
  · 改写失败（异常 / 空 / 超长 / 不可用）→ 退回原 query，绝不抛异常
"""
from types import SimpleNamespace

import pytest

import src.query_rewrite as qr


class _FakeCompletions:
    def __init__(self, calls: list, result: str, error: Exception | None):
        self._calls = calls
        self._result = result
        self._error = error

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._result))]
        )


class _FakeClient:
    """假 OpenAI 客户端：记录调用参数，按需返回内容或抛异常。"""

    def __init__(self, result: str = "改写后的问题", error: Exception | None = None):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(self.calls, result, error)
        )


HISTORY = [
    {"role": "user", "content": "混合检索是怎么做的？"},
    {"role": "assistant", "content": "BM25 + 向量 + RRF 融合。"},
]


# ---------- 不该改写 ----------

def test_no_history_returns_original():
    """单轮提问：零成本原样返回。"""
    client = _FakeClient()
    assert qr.rewrite_query("什么是RAG", None, client=client) == "什么是RAG"
    assert client.calls == []


def test_empty_history_returns_original():
    """空历史列表等同于没有历史。"""
    client = _FakeClient()
    assert qr.rewrite_query("什么是RAG", [], client=client) == "什么是RAG"
    assert client.calls == []


def test_history_without_user_message_returns_original():
    """只有助手消息（异常场景）不触发改写。"""
    client = _FakeClient()
    history = [{"role": "assistant", "content": "上一轮回答"}]
    assert qr.rewrite_query("那第二点呢", history, client=client) == "那第二点呢"
    assert client.calls == []


def test_blank_user_content_returns_original():
    """用户消息是空白字符也不算有历史。"""
    client = _FakeClient()
    history = [{"role": "user", "content": "   "}]
    assert qr.rewrite_query("那第二点呢", history, client=client) == "那第二点呢"
    assert client.calls == []


def test_disabled_switch_returns_original(monkeypatch):
    """开关关闭时即便有历史也不改写（A/B 对照用）。"""
    monkeypatch.setattr(qr, "QUERY_REWRITE_ENABLED", False)
    client = _FakeClient()
    assert qr.rewrite_query("那第二点呢", HISTORY, client=client) == "那第二点呢"
    assert client.calls == []


# ---------- 该改写 ----------

def test_rewrite_success():
    """有历史 → 返回模型的改写结果。"""
    client = _FakeClient("混合检索是怎么实现的？")
    got = qr.rewrite_query("那第二点呢", HISTORY, client=client)
    assert got == "混合检索是怎么实现的？"
    assert len(client.calls) == 1


def test_prompt_contains_history_and_system(monkeypatch):
    """提示词必须是 system + user 两条，且 user 里带上历史与当前问题。"""
    client = _FakeClient("改写后的问题")
    qr.rewrite_query("那第二点呢", HISTORY, client=client)
    messages = client.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "只补全指代" in messages[0]["content"]
    user_msg = messages[1]["content"]
    assert "混合检索是怎么做的？" in user_msg          # 历史里的用户消息
    assert "BM25 + 向量 + RRF 融合。" in user_msg      # 历史里的助手消息
    assert "那第二点呢" in user_msg                    # 当前问题
    assert client.calls[0]["temperature"] == 0.0


def test_temperature_zero_and_model_passthrough():
    """改写要确定性，且 model 可注入。"""
    client = _FakeClient("x")
    qr.rewrite_query("追问", HISTORY, client=client, model="my-model")
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["model"] == "my-model"


def test_only_recent_history_used(monkeypatch):
    """只送最近 RAG_HISTORY_TURNS 条历史，更早的不进提示词。"""
    monkeypatch.setattr(qr, "RAG_HISTORY_TURNS", 2)
    history = [{"role": "user", "content": f"很久以前的问题{i}"} for i in range(5)]
    client = _FakeClient("改写结果")
    qr.rewrite_query("追问", history, client=client)
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "很久以前的问题4" in user_msg
    assert "很久以前的问题0" not in user_msg


def test_long_history_entry_truncated():
    """单条历史超长要截断，防止提示词膨胀。"""
    history = [{"role": "user", "content": "长" * 2000}]
    client = _FakeClient("改写结果")
    qr.rewrite_query("追问", history, client=client)
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "长" * qr._HISTORY_CHAR_LIMIT in user_msg
    assert "长" * (qr._HISTORY_CHAR_LIMIT + 1) not in user_msg


# ---------- 结果清洗 ----------

def test_strips_prefix_and_quotes():
    """模型画蛇添足加前缀/引号 → 剥掉。"""
    client = _FakeClient('改写结果：“混合检索的候选池是怎么设的？”')
    got = qr.rewrite_query("那第二点呢", HISTORY, client=client)
    assert got == "混合检索的候选池是怎么设的？"


def test_takes_first_nonempty_line():
    """模型多输出一行解释 → 只取首个非空行。"""
    client = _FakeClient("\n\n混合检索的候选池怎么设\n（补充说明）")
    got = qr.rewrite_query("那第二点呢", HISTORY, client=client)
    assert got == "混合检索的候选池怎么设"


# ---------- 失败降级 ----------

def test_llm_exception_falls_back(capsys):
    """模型调用异常 → 退回原 query，不冒泡。"""
    client = _FakeClient(error=RuntimeError("boom"))
    got = qr.rewrite_query("那第二点呢", HISTORY, client=client)
    assert got == "那第二点呢"
    assert "QueryRewrite" in capsys.readouterr().err


def test_empty_result_falls_back():
    """模型返回空 → 退回原 query。"""
    client = _FakeClient("")
    assert qr.rewrite_query("那第二点呢", HISTORY, client=client) == "那第二点呢"


def test_too_long_result_falls_back():
    """模型开始「答题」而不是「改写」（超长）→ 退回原 query。"""
    client = _FakeClient("这个问题的答案需要从三个方面来说明。" * 20)
    assert qr.rewrite_query("那第二点呢", HISTORY, client=client) == "那第二点呢"


def test_single_char_result_falls_back():
    """只有 1 个字符的结果视为无效。"""
    client = _FakeClient("好")
    assert qr.rewrite_query("那第二点呢", HISTORY, client=client) == "那第二点呢"


def test_none_content_falls_back():
    """message.content 为 None（部分服务端行为）→ 退回原 query。"""
    client = _FakeClient(None)
    assert qr.rewrite_query("那第二点呢", HISTORY, client=client) == "那第二点呢"


# ---------- needs_rewrite ----------

@pytest.mark.parametrize("history,expected", [
    (None, False),
    ([], False),
    ([{"role": "assistant", "content": "x"}], False),
    ([{"role": "user", "content": ""}], False),
    ([{"role": "user", "content": "问"}], True),
    ([{"role": "assistant", "content": "答"}, {"role": "user", "content": "问"}], True),
])
def test_needs_rewrite(history, expected):
    assert qr.needs_rewrite(history) is expected


def test_unknown_role_ignored():
    """历史里混入非 user/assistant 的条目不影响判定与渲染。"""
    client = _FakeClient("改写结果")
    history = [{"role": "system", "content": "系统消息"}, {"role": "user", "content": "问"}]
    got = qr.rewrite_query("追问", history, client=client)
    assert got == "改写结果"
    assert "系统消息" not in client.calls[0]["messages"][1]["content"]


def test_non_dict_history_entry_ignored():
    """历史里混入非 dict（脏数据）不崩。"""
    client = _FakeClient("改写结果")
    history = ["脏数据", {"role": "user", "content": "问"}]
    assert qr.rewrite_query("追问", history, client=client) == "改写结果"
