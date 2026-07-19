"""
dev_agent_v2.py —— 企业级改进版

和 v1 的区别：
  1. print() → logging（可控制级别、可写文件、可远程收集）
  2. 每个工具调用都有异常保护（工具挂了不影响 Agent 继续运行）
  3. 资源管理：程序退出时自动保存向量库
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

# ================================================================
# 改进 1：用 logging 代替 print
# ================================================================
# print() 的缺点：
#   - 无法区分「信息」和「错误」
#   - 无法关闭（上线后你不需要看每次工具调用）
#   - 无法写入文件（出问题后没法追溯）
#
# logging 等级：DEBUG < INFO < WARNING < ERROR < CRITICAL
# 开发时用 DEBUG，上线后用 WARNING（只记录重要的）

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dev_agent")


# ================================================================
# 改进 2：工具执行器 —— 带异常保护的统一入口
# ================================================================
# v1 的问题：
#   func(**args) 直接调用，如果函数内部报错，整个 Agent 崩溃
#   → 用户看到一堆 traceback，体验极差
#
# v2 的解决方案：
#   每个工具调用都包在 try-except 里
#   → 工具挂了返回错误信息给 AI，Agent 继续运行
#   → AI 看到错误后会尝试其他方式

def execute_tool_safely(name: str, args: dict, tool_map: dict) -> str:
    """安全执行工具 —— 企业级标配"""
    func = tool_map.get(name)
    if func is None:
        return f"错误: 未知工具 '{name}'"

    try:
        result = func(**args)
        logger.info(f"工具 {name}({args}) 执行成功")
        return result
    except Exception as e:
        logger.error(f"工具 {name}({args}) 执行失败: {str(e)}")
        # 关键：把错误信息返回给 AI，而不是崩溃
        # AI 看到这个错误后会自己决定怎么办（重试？换个方式？告诉用户？）
        return f"工具 '{name}' 执行失败: {str(e)}。请基于已有信息继续，或尝试其他方式。"


# ================================================================
# 工具配置（和 v1 一样，不变）
# ================================================================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

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
# Agent 循环（核心逻辑和 v1 一样，只是工具调用用了安全执行器）
# ================================================================

def run_agent(question: str, work_dir: str = ".", max_steps: int = 8) -> str:
    """ReAct Agent 主循环（v2 企业版）"""
    logger.info(f"收到问题: {question[:50]}...")

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个开发助手 Agent。\n"
                "规则:\n"
                "1. 需要信息时调用工具，不要猜测\n"
                "2. 基于工具返回的真实数据回答\n"
                "3. 如果工具执行失败，尝试其他方式或告知用户\n"
                f"4. 当前工作目录: {work_dir}"
            ),
        },
        {"role": "user", "content": question},
    ]

    step = 0
    tools_used = []

    while step < max_steps:
        step += 1
        logger.debug(f"Agent 步骤 {step}")

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            return "AI 服务暂时不可用，请稍后重试。"

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                logger.info(f"调用工具: {name}({args})")

                # ⭐ 改进点：用安全执行器代替直接调用
                result = execute_tool_safely(name, args, TOOL_MAP)
                tools_used.append(name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        else:
            answer = msg.content
            logger.info(f"完成: {step} 步, 用了 {len(tools_used)} 个工具")
            return answer

    logger.warning(f"达到最大步数 {max_steps}，强制结束")
    return "抱歉，思考了太久。请换个方式提问。"


# ================================================================
# 改进 3：程序退出时自动保存
# ================================================================

def cleanup():
    """程序退出时自动保存向量库"""
    store = get_vector_store()
    if store and store.is_loaded:
        from src.tools.rag_tool import VECTOR_PATH
        store.save(VECTOR_PATH)
        logger.info("向量库已保存")


atexit.register(cleanup)


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 开发助手 Agent v2（企业版）")
    print("=" * 50)
    print("v2 改进: logging 代替 print | 工具异常保护 | 退出自动保存")
    print("试试: '列一下桌面文件' / '加载 111.txt 到知识库'")
    print("输入 'exit' 退出")
    print("=" * 50)

    while True:
        q = input("\n👤 你: ")
        if q.lower() in ("exit", "quit", "q"):
            break
        answer = run_agent(q, work_dir="C:/Users/24162/Desktop")
        print(f"\n🤖 {answer}")
        print("-" * 50)
