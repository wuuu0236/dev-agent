"""
Agent API 服务 —— 把 dev_agent 变成 HTTP 接口

v2 改动：/chat 接口从 while 循环换成 LangGraph 图
  · 以前：server.py 自己写 while 循环调 AI
  · 现在：server.py 只管接收请求 → 调 graph → 返回结果
  · Agent 的核心逻辑全在 dev_agent_langgraph.py 的图里

跑起来后：
  浏览器访问 http://localhost:8000/docs → 自动生成的 API 文档
  POST http://localhost:8000/chat → 发问题，拿回答
"""

import os
import sys
import asyncio
import logging
import io
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ⭐ 统一入口：跑 LangGraph 图 + 提取最终答案，都在 run_agent 里
from src.agent.dev_agent_langgraph import run_agent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent_api")


# ================================================================
# FastAPI 应用
# ================================================================

app = FastAPI(
    title="开发助手 Agent API",
    description="文件操作 + 知识库问答 —— 把 dev_agent 变成 HTTP 服务",
    version="1.0.0",
    # 自定义 Swagger 文档页面
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,   # 隐藏底部的 Schema（减少英文干扰）
        "displayRequestDuration": True,   # 显示请求耗时
        "docExpansion": "list",           # 接口默认展开
    },
)


# 请求/响应模型（FastAPI 自动校验 + 生成文档）
class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = Field(
        description="你想问的问题，例如：'列出桌面的文件'、'读取 111.txt 的内容'",
        examples=["列出桌面的所有文件"],
    )
    work_dir: str = Field(
        default="C:/Users/24162/Desktop",
        description="工作目录，Agent 只能访问该目录下的文件",
    )


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(description="Agent 的回答内容")
    steps: int = Field(description="Agent 执行了多少步（调了几次工具）")
    timestamp: str = Field(description="响应时间")


# ================================================================
# API 端点
# ================================================================

@app.get(
    "/",
    summary="首页",
    description="返回服务基本信息",
)
def root():
    """查看服务是否在运行"""
    return {
        "服务": "开发助手 Agent API",
        "版本": "1.0.0",
        "文档地址": "/docs",
    }


@app.get(
    "/health",
    summary="健康检查",
    description="检查服务是否正常运行",
)
def health():
    """健康检查接口"""
    return {"状态": "正常", "时间": datetime.now().isoformat()}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="发送问题（LangGraph 版）",
    description="发送一个问题，Agent 自动调用工具后返回答案。底层是 LangGraph 图。",
)
def chat(request: ChatRequest):
    """
    向 Agent 提问

    流程：
      收到 HTTP 请求 → run_agent() 跑图 → 返回 {answer, steps}

    server.py 不再管「怎么调 AI」「怎么调工具」「怎么提取最终答案」，
    全部收敛到 dev_agent_langgraph.py 的 run_agent()。
    """
    logger.info(f"收到问题: {request.question[:50]}...")

    result = run_agent(request.question, work_dir=request.work_dir, session_id="api")

    return ChatResponse(
        answer=result["answer"],
        steps=result["steps"],
        timestamp=datetime.now().isoformat(),
    )


@app.post(
    "/chat/stream",
    summary="发送问题（流式版）",
    description="和 /chat 一样，但最终答案会分块流式返回（打字机效果）。",
)
async def chat_stream(request: ChatRequest):
    """
    向 Agent 提问，答案流式输出

    图只跑一次（和 /chat 开销相同），拿到最终答案后按小块吐给前端。
    旧版会在图跑完后【再调一次 LLM 流式接口】——费用翻倍，且两次生成的
    答案可能不一致。现在答案以 /chat 为准，分块输出只负责「打字机效果」。
    """
    async def generate():
        result = run_agent(request.question, work_dir=request.work_dir, session_id="api-stream")
        answer = result["answer"]

        # 分块输出：块太大等于一次返回，块太小切碎中文读感差
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            await asyncio.sleep(0.03)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# ================================================================
# 启动
# ================================================================

if __name__ == "__main__":
    import uvicorn

    # 解决 Windows 终端 emoji 编码问题
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 50)
    print("[Dev Agent API] 启动成功!")
    print("=" * 50)
    print("API 文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("测试命令: curl -X POST http://localhost:8000/chat")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
