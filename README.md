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
│   │   ├── safety.py                 # 工具安全审查（黑/敏感/白名单）
│   │   ├── rag_tool.py               # 向量知识库 + RAG 检索（纯稠密向量）
│   │   └── hybrid_retriever.py       # 混合检索：BM25 + 稠密向量 + RRF 融合
│   ├── api/
│   │   └── server.py                 # FastAPI 服务（集成 LangGraph）
│   ├── mcp_server.py                 # MCP 协议工具服务器
│   └── evaluate_rag.py               # RAGAS 评估脚本（4 组对照实验）
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

## Agent 工具

| 工具 | 类型 | 说明 |
|------|------|------|
| `list_files` | 文件 | 列出目录下的文件和文件夹 |
| `read_file` | 文件 | 读取文件内容（含安全检查） |
| `search_in_files` | 文件 | 按关键词搜索文件 |
| `search_knowledge` | RAG | 混合检索知识库（BM25 + 稠密 + RRF） |
| `add_knowledge` | RAG | 添加文本到知识库 |
| `load_file_to_knowledge` | RAG | 加载文件到知识库 |

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM | DeepSeek API |
| Agent 框架 | LangGraph (StateGraph) |
| API | FastAPI + Uvicorn |
| 检索 | BM25 + 稠密向量 + RRF 混合检索（自实现） |
| 安全 | 三层审查：黑名单 + 敏感文件 + 路径白名单 |
| 评估 | RAGAS 四指标 + 4 组超参对照实验 |
| MCP | FastMCP |
| 部署 | Docker + Docker Compose |

---

## RAG 管线

```
用户提问 → BM25 关键词召回 → RRF 融合 → [可选] Cross-Encoder 精排 → LLM 生成
          → 稠密向量语义召回 ↗
```

- **双路召回**：BM25 精确匹配 + 稠密向量语义匹配，互补短板
- **RRF 融合**：倒数排名融合，解决关键词分数和语义分数量纲不同的问题
- **RAGAS 评估**：自建 20 条测试集，4 组超参对照实验，Precision 从 64% 提升至 87%

---

## 已知问题

- Python 3.13 与 sentence-transformers 存在兼容问题，本地需 Python 3.11
- Docker 环境统一用 Python 3.11，RAG 功能正常

---

## 许可证

MIT
