"""
开发助手 Agent —— 能读代码、搜文件、分析项目

这是应用层 Agent 的标准写法：
  调 API（不是训练） + 定义工具（不是算法） + 控制循环（工程逻辑）

使用方法：
  python -m src.agent.dev_agent
  或在项目根目录：python src/agent/dev_agent.py
"""

import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv

# 把项目根目录加到 sys.path，方便导入 tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tools.file_tools import list_files, read_file, search_in_files
from src.tools.rag_tool import search_knowledge, load_file_to_knowledge, add_knowledge

load_dotenv()

# ================================================================
# 第 1 步：创建 LLM 客户端（不是训练，是调 API！）
# ================================================================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ================================================================
# 第 2 步：定义工具的 JSON Schema（给 AI 看的"说明书"）
# ================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出指定目录下的所有文件和文件夹",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "要查看的目录路径，如 '.' 表示当前目录",
                    }
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。如果文件太长会自动只返回开头和结尾。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "要读取的文件路径",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "在目录中搜索包含指定关键词的文件，返回文件列表和出现次数",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "要搜索的目录路径",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "要搜索的关键词",
                    },
                },
                "required": ["directory", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "语义搜索知识库。当用户问知识库相关问题时使用，比关键词搜索更智能。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，用自然语言描述想找什么",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_file_to_knowledge",
            "description": "把文件加载到知识库。当用户让你'记住'或'学习'某个文件时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "要加载的文件路径",
                    }
                },
                "required": ["filepath"],
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
                    }
                },
                "required": ["texts"],
            },
        },
    },
]

# 工具名 → 实际函数映射（应用层代码，和算法无关）
TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "search_in_files": search_in_files,
    "search_knowledge": search_knowledge,
    "load_file_to_knowledge": load_file_to_knowledge,
    "add_knowledge": add_knowledge,
}


# ================================================================
# 第 3 步：Agent 循环（这是你需要彻底理解的 30 行代码）
# ================================================================

def run_agent(question: str, work_dir: str = ".", max_steps: int = 8, verbose: bool = True):
    """
    ReAct Agent 主循环

    每次循环：
      ① 把 messages 发给 AI
      ② AI 决定：直接回答？还是调用工具？
      ③ 如果调工具 → 执行 → 结果塞回 messages → 回到①
      ④ 如果直接回答 → 返回答案，结束
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个开发助手 Agent，可以操作本地文件系统。\n"
                "规则：\n"
                "1. 需要查看文件时先调用工具，不要猜测文件内容\n"
                "2. 基于工具返回的真实数据回答问题\n"
                "3. 回答简洁、结构化\n"
                f"4. 当前工作目录: {work_dir}"
            ),
        },
        {"role": "user", "content": question},
    ]

    step = 0
    tools_used = []

    while step < max_steps:
        step += 1

        if verbose:
            print(f"\n--- 步骤 {step} ---")

        # ① 调 API（不是训练，只是一次 HTTP 请求！）
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # ② AI 想调工具？
        if msg.tool_calls:
            # 记录 AI 的工具调用请求
            messages.append(
                {
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
                }
            )

            # ③ 执行每个工具
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"  🔧 {name}({args})")

                func = TOOL_MAP.get(name)
                if func:
                    result = func(**args)
                else:
                    result = f"未知工具: {name}"

                if verbose:
                    preview = result[:100].replace("\n", " ")
                    print(f"  📋 {preview}...")

                tools_used.append(name)

                # ④ 工具结果塞回 messages（关键！不放回去 AI 就失忆了）
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        else:
            # ⑤ AI 直接回答 → 结束
            answer = msg.content

            if verbose:
                print(f"  ✅ 完成 ({step} 步, 用了 {len(tools_used)} 次工具)")
                print(f"\n{'=' * 50}")
                print(f"🤖 {answer}")
                print(f"{'=' * 50}")

            return answer

    return "抱歉，思考了太久，请换个方式提问。"


# ================================================================
# 第 4 步：交互入口
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 开发助手 Agent")
    print("=" * 50)
    print("能做的事情: 查看目录、读取文件、搜索代码关键字、语义搜索知识库")
    print("试试这些:")
    print("  '列出当前目录的文件'")
    print("  '搜索所有包含 weather 的文件'")
    print("  '加载 111.txt 到知识库'")
    print("  '知识库里有什么关于学习的内容'")
    print("输入 'exit' 退出")
    print("=" * 50)

    while True:
        q = input("\n👤 你: ")
        if q.lower() in ("exit", "quit", "q"):
            print("👋 再见！")
            break
        run_agent(q, work_dir="C:/Users/24162/Desktop")
