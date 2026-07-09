"""
MCP 服务器 —— 把 dev-agent 的工具暴露给任何 MCP 兼容的 AI

启动方式（Claude Code 会自动调，不需要手动跑）：
  配置在 settings.json 的 mcpServers 里
"""

import os
import sys
import logging

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_server")

# 尝试导入 RAG 工具（你的知识库）
try:
    from src.tools.hybrid_retriever import search_knowledge, add_knowledge, load_file_to_knowledge
    _HAS_RAG = True
    logger.info("RAG 工具加载成功")
except Exception as e:
    _HAS_RAG = False
    logger.warning(f"RAG 工具加载失败（sentence-transformers 版本问题）: {e}")

mcp = FastMCP("dev-agent-tools")


# ================================================================
# 基础工具（不依赖任何第三方库，确保能连上）
# ================================================================

@mcp.tool()
def hello() -> str:
    """测试工具：返回问候信息。用来验证 MCP 连接是否正常。"""
    return "MCP 连接成功！你的 dev-agent 工具服务器正在运行。"


@mcp.tool()
def search_user_knowledge(query: str) -> str:
    """搜索用户的知识库（向量语义搜索）。
    当用户问「知识库里有什么」「之前学了什么」「Agent 是什么」时使用。"""
    if not _HAS_RAG:
        return "知识库暂不可用。sentence-transformers 与 Python 3.13 有兼容问题，需要等待库升级。"
    return search_knowledge(query)


@mcp.tool()
def add_user_knowledge(texts: list[str]) -> str:
    """把文本添加到用户的知识库。当用户说「记住」「存下来」时使用。"""
    if not _HAS_RAG:
        return "知识库暂不可用。sentence-transformers 与 Python 3.13 有兼容问题。"
    return add_knowledge(texts)


@mcp.tool()
def load_file_to_user_knowledge(filepath: str) -> str:
    """把文件加载到知识库。当用户说「学习这个文件」「记住这个文件」时使用。"""
    if not _HAS_RAG:
        return "知识库暂不可用。sentence-transformers 与 Python 3.13 有兼容问题。"
    return load_file_to_knowledge(filepath)


if __name__ == "__main__":
    print("=" * 50)
    print("[MCP Server] 开发助手工具服务器")
    print("=" * 50)
    print(f"RAG 工具: {'可用' if _HAS_RAG else '暂不可用（依赖版本问题）'}")
    print("已注册工具:")
    print("  - hello")
    print("  - search_user_knowledge")
    print("  - add_user_knowledge")
    print("  - load_file_to_user_knowledge")
    print("=" * 50)
    mcp.run()
