"""
dev_agent_v3.py —— 流式输出版

和 v2 的区别：
  只有一点：API 调用加了 stream=True
  效果：AI 回答一个字一个字往外蹦，用户不用干等
"""

import os
import json
import sys
import logging
import atexit
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.tools.file_tools import list_files, read_file, search_in_files
from src.tools.rag_tool import get_vector_store
from src.tools.hybrid_retriever import search_knowledge, load_file_to_knowledge, add_knowledge

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dev_agent")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def execute_tool_safely(name: str, args: dict, tool_map: dict) -> str:
    func = tool_map.get(name)
    if func is None:
        return f"错误: 未知工具 '{name}'"
    try:
        result = func(**args)
        logger.info(f"工具 {name} 执行成功")
        return result
    except Exception as e:
        logger.error(f"工具 {name} 执行失败: {str(e)}")
        return f"工具 '{name}' 执行失败: {str(e)}。请基于已有信息继续。"


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
            "description": "读取文件内容",
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
            "description": "在目录中搜索包含关键词的文件",
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
            "description": "语义搜索知识库",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索查询"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_file_to_knowledge",
            "description": "把文件加载到知识库",
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
            "name": "add_knowledge",
            "description": "把文本添加到知识库",
            "parameters": {
                "type": "object",
                "properties": {"texts": {
                    "type": "array", "items": {"type": "string"},
                    "description": "文本列表",
                }},
                "required": ["texts"],
            },
        },
    },
]

TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "search_in_files": search_in_files,
    "search_knowledge": search_knowledge,
    "load_file_to_knowledge": load_file_to_knowledge,
    "add_knowledge": add_knowledge,
}


# ================================================================
# ⭐ v3 的核心变化：流式输出最终回答
# ================================================================
# v2: response = client.chat.completions.create(...)
#      → 一次性返回全部回答，用户干等 5-10 秒
#
# v3: stream = client.chat.completions.create(..., stream=True)
#      → 一个字一个字返回，像 ChatGPT 一样打字效果
#
# ⚠️ 注意：流式只用于「最终回答」。
# 工具调用阶段不能用流式（因为需要完整的 tool_calls 才能解析）。


def run_agent(question: str, work_dir: str = ".", max_steps: int = 8) -> str:
    """ReAct Agent（v3 流式版）"""
    logger.info(f"收到问题: {question[:50]}...")

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个开发助手 Agent。\n"
                "规则: 需要信息时调用工具，基于真实数据回答，回答简洁。\n"
                f"当前工作目录: {work_dir}"
            ),
        },
        {"role": "user", "content": question},
    ]

    step = 0

    while step < max_steps:
        step += 1
        logger.debug(f"步骤 {step}")

        # 工具调用阶段：不用流式（需要完整 tool_calls）
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                logger.info(f"调用工具: {name}")

                result = execute_tool_safely(name, args, TOOL_MAP)

                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })

        else:
            # ====================================================
            # ⭐ 最终回答阶段：用流式输出！
            # ====================================================
            print()  # 换行
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,       # ← 这就是唯一的改动
            )

            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)  # 一个字一个字打印
                    full_answer += text

            print()  # 最后的换行
            logger.info(f"完成: {step} 步")
            return full_answer

    return "抱歉，思考了太久。"


def cleanup():
    store = get_vector_store()
    if store and store.is_loaded:
        from src.tools.rag_tool import VECTOR_PATH
        store.save(VECTOR_PATH)


atexit.register(cleanup)


if __name__ == "__main__":
    print("=" * 50)
    print("🤖 开发助手 Agent v3（流式版）")
    print("=" * 50)
    print("v3 改进: 最终回答流式输出（打字机效果）")
    print("输入 'exit' 退出")
    print("=" * 50)

    while True:
        q = input("\n👤 你: ")
        if q.lower() in ("exit", "quit", "q"):
            break
        run_agent(q, work_dir="C:/Users/24162/Desktop")
