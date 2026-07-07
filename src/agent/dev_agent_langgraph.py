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
import json
import logging
from typing import TypedDict, Annotated, Literal

from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tools.file_tools import list_files, read_file, search_in_files

# RAG 工具：可选（需要 sentence-transformers 且第一次要下载模型）
# 暂时注释掉，不然网络不好会卡死
# from src.tools.rag_tool import search_knowledge, load_file_to_knowledge, add_knowledge

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("langgraph_agent")

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

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 工具定义（暂时只用文件工具，RAG 先注释）
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
]

TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "search_in_files": search_in_files,
}


# ── 格式转换层 ──────────────────────────────────────────
# LangChain 消息  ←→  OpenAI API 格式
# 两边字段名和值都不一样，需要翻译

def _langchain_to_openai(messages: list) -> list:
    """LangChain 消息 → OpenAI API 格式"""
    TYPE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}

    openai_msgs = []
    for msg in messages:
        d = msg.model_dump()
        lc_type = d.get("type", "")
        role = TYPE_MAP.get(lc_type, lc_type)
        content = d.get("content", "")

        converted = {"role": role, "content": content}

        # AI 消息里的 tool_calls：LC 格式 → OpenAI 格式
        if "tool_calls" in d and d["tool_calls"]:
            openai_tool_calls = []
            for tc in d["tool_calls"]:
                openai_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                })
            converted["tool_calls"] = openai_tool_calls

        # ToolMessage 需要 tool_call_id，AI 才知道对应哪个工具调用
        if lc_type == "tool" and "tool_call_id" in d:
            converted["tool_call_id"] = d["tool_call_id"]

        openai_msgs.append(converted)
    return openai_msgs


# ── 节点函数 ────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """
    节点 1：调用 AI
    ──────────────
    ① 把 LangChain 消息转成 OpenAI 格式
    ② 调 DeepSeek API
    ③ 把 OpenAI 回复转回 LangChain 格式

    对应 while 循环：response = client.chat.completions.create(...)
    """
    logger.info(f"节点 call_model：调用 AI（第 {state['step_count']} 步）...")

    openai_messages = _langchain_to_openai(state["messages"])

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=openai_messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    from langchain_core.messages import AIMessage
    msg = response.choices[0].message

    # tool_calls 格式转换：OpenAI → LangChain
    lc_tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            lc_tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
            })

    ai_msg = AIMessage(content=msg.content or "", tool_calls=lc_tool_calls)
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

def run_agent(question: str, work_dir: str = "C:/Users/24162/Desktop"):
    """运行 LangGraph Agent（使用方式和 v3 一样）"""
    from langchain_core.messages import HumanMessage, SystemMessage

    initial_state = {
        "messages": [
            SystemMessage(content=f"你是开发助手 Agent。当前工作目录: {work_dir}"),
            HumanMessage(content=question),
        ],
        "step_count": 0,
    }

    logger.info(f"收到问题: {question[:50]}...")

    # graph.stream() 每次 yield 一个节点执行的结果
    for event in graph.stream(initial_state):
        node_name = list(event.keys())[0]
        node_data = event[node_name]
        if "messages" in node_data:
            msgs = node_data["messages"]
            last_msg = msgs[-1]
            preview = str(getattr(last_msg, 'content', ''))[:80].replace('\n', ' ')
            logger.debug(f"[{node_name}] {preview}")

    # 获取最终状态中的最后一条消息（AI 回答）
    final_state = event[list(event.keys())[0]] if event else {}
    return "抱歉，Agent 没有生成回答。"


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("[LangGraph] 开发助手 Agent")
    print("=" * 50)
    print("和 v3 功能一样，但用 StateGraph 代替 while 循环")
    print("当前工具: list_files | read_file | search_in_files")
    print("输入 'exit' 退出")
    print("=" * 50)

    while True:
        q = input("\n你: ")
        if q.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        # run_agent 现在只打日志，不打印中间过程
        # 最终回答在下面手动输出
        from langchain_core.messages import HumanMessage, SystemMessage

        initial_state = {
            "messages": [
                SystemMessage(content=f"你是开发助手 Agent。工作目录: C:/Users/24162/Desktop"),
                HumanMessage(content=q),
            ],
            "step_count": 0,
        }

        final_answer = ""
        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            if node_name == "call_model":
                msgs = event["call_model"].get("messages", [])
                if msgs:
                    last = msgs[-1]
                    # 如果 AI 直接回答了（没有 tool_calls），就是最终答案
                    if not getattr(last, 'tool_calls', None):
                        final_answer = getattr(last, 'content', str(last))

        if not final_answer:
            final_answer = "抱歉，Agent 没有生成回答。"

        print(f"\n[Agent] {final_answer}")
        print("-" * 50)
