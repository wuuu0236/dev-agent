"""
dev_agent_langgraph.py —— LangGraph 版 Agent

和 v3 功能完全一样（ReAct：思考 → 调工具 → 思考 → 回答）
区别：用 StateGraph（图）代替 while（循环）

三个核心概念：
  1. State  — 在图中流动的数据（类比快递包裹）
  2. Node   — 处理 State 的函数（类比流水线工人）
  3. Edge   — 节点之间的连线（类比岔路口指示牌）
"""

import os
import sys
import logging
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tools.file_tools import list_files, read_file, search_in_files
from src.hybrid_retriever import search_knowledge, add_knowledge, load_file_to_knowledge
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, RAG_HISTORY_TURNS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("langgraph_agent")

# --- Langfuse 观测层（可选）---
# 每一次 graph.stream() 挂上 CallbackHandler，
# Langfuse 面板会自动记录 call_model / call_tools 两个节点的完整 span。
# 缺 langchain 包或缺 Langfuse 密钥时降级为不观测，Agent 主流程不受影响。
langfuse_handler = None
try:
    from langfuse.callback.langchain import LangchainCallbackHandler
    langfuse_handler = LangchainCallbackHandler()  # 自动从 LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST 环境变量读取
except Exception as e:
    logger.warning(f"Langfuse 不可用，本次会话不观测: {e}")

# ================================================================
# 第 1 步：定义 State（图中流动的数据）
# ================================================================
# while 循环的 State 就是 messages + step 计数器
# LangGraph 把它定义成一个明确的类型

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # ↑ Annotated[list, add_messages]：每次节点返回新消息时，自动追加到列表
    step_count: int
    # ↑ 步数计数器，达到上限就强制结束（和 while step < max_steps 一样）


# ================================================================
# 第 2 步：定义节点
# ================================================================

# 用 LangChain 的 ChatOpenAI（原生吃 LangChain 消息，不再需要手动格式转换）。
# 正因为是 LangChain 聊天模型，graph.stream(stream_mode="messages") 才能按 token 真流式。
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=LLM_TEMPERATURE,
)

# 工具定义：文件工具 + 知识库（search_knowledge 等 RAG 工具已启用）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出指定目录下的所有文件和文件夹",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string", "description": "目录路径"}},
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。如果文件太长会自动截断。",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "文件路径"}},
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "在目录中搜索包含指定关键词的文件，返回文件列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径"},
                    "keyword": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["directory", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库。当用户问 AI/编程/技术相关问题时使用，支持中英文混合查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询，用自然语言描述想找什么"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_knowledge",
            "description": "把文本添加到知识库。当用户直接告诉你一些信息需要记住时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要添加的文本列表",
                    },
                },
                "required": ["texts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_file_to_knowledge",
            "description": "把文件加载到知识库。当用户让你记住或学习某个文件时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "要加载的文件路径"},
                },
                "required": ["filepath"],
            },
        },
    },
]

TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "search_in_files": search_in_files,
    "search_knowledge": search_knowledge,
    "add_knowledge": add_knowledge,
    "load_file_to_knowledge": load_file_to_knowledge,
}

# 绑好工具的模型：invoke(LangChain 消息) → AIMessage，tool_calls 直接是 LangChain 格式
llm_with_tools = llm.bind_tools(TOOLS)


# ── 格式转换说明 ────────────────────────────────────────
# 旧版（v3）：call_model 用裸 OpenAI 客户端，需要手动把 LangChain 消息
#   翻译成 OpenAI API 格式（曾有一个 _langchain_to_openai 转换层）。
# 现在：直接用 LangChain 的 ChatOpenAI 模型，它原生接受 LangChain 消息，
#   tool_calls 也直接返回 LangChain 格式——转换层删掉，少一层出错面。


# ── 节点函数 ────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """
    节点 1：调用 AI
    ──────────────
    绑了工具的 ChatOpenAI 直接 invoke LangChain 消息，返回 AIMessage。

    对应 while 循环：response = client.chat.completions.create(...)
    """
    logger.info(f"节点 call_model：调用 AI（第 {state['step_count']} 步）...")

    # 绑了工具的 ChatOpenAI：直接 invoke LangChain 消息，
    # 返回的 AIMessage 自带 LangChain 格式的 tool_calls（id / name / args）
    ai_msg = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [ai_msg],
        "step_count": state["step_count"] + 1,  # 每次调 AI 就 +1（= while 循环的 step += 1）
    }


def call_tools(state: AgentState) -> dict:
    """
    节点 2：执行工具
    ──────────────
    读取 AI 要求的 tool_calls，逐个执行，返回结果

    对应 while 循环：for tc in msg.tool_calls: func(**args)
    """
    ai_message = state["messages"][-1]

    from langchain_core.messages import ToolMessage

    tool_results = []
    for tc in ai_message.tool_calls:
        name = tc["name"]
        args = tc["args"]

        logger.info(f"节点 call_tools：执行 {name}({args})")

        func = TOOL_MAP.get(name)
        try:
            result = func(**args)
        except Exception as e:
            result = f"工具 '{name}' 执行失败: {str(e)}"
            logger.error(result)

        tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {"messages": tool_results}


# ================================================================
# 第 3 步：定义边（岔路口判断）
# ================================================================

MAX_STEPS = 8  # 最大步数，和 while 循环的 max_steps 一样

def should_continue(state: AgentState) -> Literal["call_tools", "__end__"]:
    """
    决定下一步去哪：
      ① 有 tool_calls → 去 call_tools
      ② 没有 tool_calls → 结束（AI 直接回答了）
      ③ 步数超限 → 强制结束

    对应 while 循环：if msg.tool_calls: ... else: ...
    """
    # 步数超限 → 强制结束
    if state["step_count"] >= MAX_STEPS:
        logger.warning(f"步数达到上限 {MAX_STEPS}，强制结束")
        return "__end__"

    ai_message = state["messages"][-1]

    if ai_message.tool_calls:
        logger.info(f"判断：AI 要调工具 → 去 call_tools（第 {state['step_count']} 步）")
        return "call_tools"
    else:
        logger.info(f"判断：AI 直接回答 → 结束（共 {state['step_count']} 步）")
        return "__end__"


# ================================================================
# 第 4 步：搭图
# ================================================================

workflow = StateGraph(AgentState)

workflow.add_node("call_model", call_model)
workflow.add_node("call_tools", call_tools)

# 入口：先从 0 开始，然后进入循环
workflow.set_entry_point("call_model")

# call_model 之后 → 判断要不要调工具
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {"call_tools": "call_tools", "__end__": END},
)

# 工具调完 → 先计数 → 回到 call_model（形成循环）
workflow.add_edge("call_tools", "call_model")

graph = workflow.compile()


# ================================================================
# 第 5 步：运行
# ================================================================

SYSTEM_PROMPT = """你是开发助手 Agent，当前工作目录: {work_dir}。你可以调用工具完成文件操作，也可以查询知识库回答问题。

规则：
1. 需要文件/工具操作时，先调用对应工具再回答；工具执行失败要如实说明，不要假装成功。
2. 查询知识库（search_knowledge）时，工具结果会自带 [n] 来源标注。引用时沿用这些序号，格式 [n]，不要编造文件名或页码。
3. 用中文简洁回答。"""


def _build_initial_state(question: str, work_dir: str,
                         history: list[dict] | None = None) -> dict:
    """组装图的初始状态：system + 最近对话历史 + 当前问题。

    history 约定：此条问题之前的对话，[{role, content}]，最近的在列表末尾；
    和 RAG 问答路径一致（网页注入 messages[:-1]）。只取最近 RAG_HISTORY_TURNS 轮，
    让「那第二点呢」「具体怎么配置」这类追问有上下文。
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(work_dir=work_dir))]
    if history:
        for h in history[-RAG_HISTORY_TURNS:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=question))
    return {"messages": messages, "step_count": 0}


def run_agent(question: str, work_dir: str = "C:/Users/24162/Desktop",
              session_id: str = "api",
              history: list[dict] | None = None) -> dict:
    """
    运行 LangGraph Agent，返回 {"answer": str, "steps": int}。

    「跑图 → 提取最终答案」只在这里做一遍：
      · API 的 /chat 走这里
      · CLI 交互走这里
    /chat/stream 走 stream_agent（同一个图，按 token 吐）。
    history：此条问题之前的对话，[{role, content}]，多轮追问有上下文。
    """
    initial_state = _build_initial_state(question, work_dir, history)

    logger.info(f"收到问题: {question[:50]}...")

    final_answer = ""
    final_step = 0
    stream_config: dict = {
        "metadata": {"session_id": session_id, "user_question": question[:80]},
        "run_name": "dev-agent-langgraph",
    }
    if langfuse_handler is not None:
        stream_config["callbacks"] = [langfuse_handler]

    for event in graph.stream(initial_state, config=stream_config):
        node_name = list(event.keys())[0]
        node_data = event[node_name]

        # 捕获最终答案：call_model 节点且没有 tool_calls 时，就是 AI 的直接回答
        # （有 tool_calls 的消息 content 通常为空，会被天然跳过）
        if node_name == "call_model":
            msgs = node_data.get("messages", [])
            final_step = node_data.get("step_count", final_step)
            if msgs:
                last = msgs[-1]
                if not getattr(last, 'tool_calls', None):
                    final_answer = getattr(last, 'content', str(last))

    return {
        "answer": final_answer or "抱歉，Agent 没有生成回答。",
        "steps": final_step,
    }


def stream_agent(question: str, work_dir: str = "C:/Users/24162/Desktop",
                 session_id: str = "api-stream",
                 history: list[dict] | None = None):
    """流式版 run_agent：同一个图、同一个 LLM 调用，按 token 吐最终答案。

    关键在 call_model 用了 LangChain 聊天模型，配合
    graph.stream(stream_mode=["updates", "messages"]) 能拿到每个 token：
      · updates 模式 → 每个节点跑完的结果（这里忽略）
      · messages 模式 → (AIMessageChunk, metadata)，chunk.content 就是新 token
    工具调用的中间消息 content 为空，自动跳过——只有最终答案的文字流出来，
    所以这是真流式，不是「跑完再切块」（旧版假流式，还白等一整轮）。
    """
    initial_state = _build_initial_state(question, work_dir, history)

    logger.info(f"收到问题（流式）: {question[:50]}...")

    stream_config: dict = {
        "metadata": {"session_id": session_id, "user_question": question[:80]},
        "run_name": "dev-agent-langgraph-stream",
    }
    if langfuse_handler is not None:
        stream_config["callbacks"] = [langfuse_handler]

    for mode, data in graph.stream(initial_state, config=stream_config,
                                   stream_mode=["updates", "messages"]):
        if mode == "messages":
            message_chunk, _meta = data
            if message_chunk.content:
                yield message_chunk.content


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("[LangGraph] 开发助手 Agent")
    print("=" * 50)
    print("和 v3 功能一样，但用 StateGraph 代替 while 循环")
    print("当前工具: list_files | read_file | search_in_files | search_knowledge")
    print("输入 'exit' 退出")
    print("=" * 50)

    while True:
        q = input("\n你: ")
        if q.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        result = run_agent(q, work_dir="C:/Users/24162/Desktop", session_id="cli-interactive")
        print(f"\n[Agent] {result['answer']}")
        print("-" * 50)
