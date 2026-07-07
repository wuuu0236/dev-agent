# Dev Agent —— 开发助手 AI Agent

基于 **LangGraph + FastAPI** 的 AI Agent，能操作本地文件、搜索知识库。支持 HTTP 调用和 Docker 一键部署。

> 从 `while` 循环到 `StateGraph`，完整记录了 Agent 架构的演进过程。

---

## 项目结构

```
dev-agent/
├── src/
│   ├── agent/
│   │   ├── dev_agent.py              # v1: 基础 Agent 循环（ReAct 模式）
│   │   ├── dev_agent_v2.py           # v2: +logging +异常保护
│   │   ├── dev_agent_v3.py           # v3: +流式输出
│   │   └── dev_agent_langgraph.py    # v4: LangGraph StateGraph
│   ├── tools/
│   │   ├── file_tools.py             # 文件工具（list / read / search）
│   │   └── rag_tool.py               # 向量知识库 + RAG 检索
│   ├── api/
│   │   └── server.py                 # FastAPI 服务（集成 LangGraph）
│   └── mcp_server.py                 # MCP 协议工具服务器
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 快速开始

### 方式一：本地运行

```bash
pip install -r requirements.txt
python src/api/server.py
# 浏览器打开 http://localhost:8000/docs
```

### 方式二：Docker 一键部署

```bash
docker compose up
# 浏览器打开 http://localhost:8000/docs
```

---

## API 使用

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "列出桌面的文件", "work_dir": "/app/host-desktop"}'
```

---

## 架构演进

| 版本 | 改进 | 解决的问题 |
|:---:|------|------|
| v1 | Agent 基础循环 | ReAct：思考 → 调工具 → 回答 |
| v2 | logging + 异常保护 | 工具崩溃不连累 Agent |
| v3 | 流式输出 | 打字机效果，不干等 |
| v4 | LangGraph StateGraph | 流程可视化，加功能加节点即可 |

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM | DeepSeek API |
| Agent 框架 | LangGraph (StateGraph) |
| API | FastAPI + Uvicorn |
| 向量检索 | Sentence-Transformers + NumPy（自实现） |
| MCP | FastMCP |
| 部署 | Docker + Docker Compose |

---

## 已知问题

- Python 3.13 与 sentence-transformers 存在兼容问题，本地需 Python 3.11
- Docker 环境统一用 Python 3.11，RAG 功能正常

---

## 许可证

MIT
