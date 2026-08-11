"""确定性测试：LangGraph Agent 的对话历史注入（不调 LLM，无需任何密钥）。

测的是 run_agent 组装初始状态那一步（_build_initial_state）：
多轮追问的上下文怎么进图，纯逻辑，可离线验证。
"""
from src.agent.dev_agent_langgraph import _build_initial_state
from src.config import RAG_HISTORY_TURNS


def test_history_injection_order():
    """system → 历史(user/assistant) → 当前问题，顺序必须对。"""
    msgs = _build_initial_state("当前问题", "C:/工作目录", history=[
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
    ])
    types = [m.type for m in msgs["messages"]]
    assert types == ["system", "human", "ai", "human"]
    assert msgs["messages"][-1].content == "当前问题"


def test_history_filters_meta_and_empty():
    """system 角色、空白内容不应被注入历史（防污染上下文）。"""
    msgs = _build_initial_state("Q", "C:/", history=[
        {"role": "system", "content": "不应注入"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "真实问题"},
    ])
    contents = [m.content for m in msgs["messages"]]
    assert "不应注入" not in contents
    assert "真实问题" in contents


def test_history_truncated_to_latest():
    """只注入最近 RAG_HISTORY_TURNS 条（与 RAG 问答路径一致的语义，按条数切）。"""
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(20)]
    msgs = _build_initial_state("Q", "C:/", history=hist)
    injected = msgs["messages"][1:-1]  # 去掉 system 和当前问题
    assert len(injected) == RAG_HISTORY_TURNS
    assert injected[0].content == str(20 - RAG_HISTORY_TURNS)  # 最新几条开头
    assert injected[-1].content == "19"


def test_system_prompt_contains_workdir():
    """system 提示里带当前工作目录（Agent 知道自己能碰哪些文件）。"""
    msgs = _build_initial_state("Q", "C:/我的工作目录", history=None)
    assert "C:/我的工作目录" in msgs["messages"][0].content
