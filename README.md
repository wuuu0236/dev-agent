# Dev Agent —— 开发助手 AI Agent + 知识库问答平台

基于 **LangGraph + FastAPI + Streamlit** 的完整 RAG 系统。支持 HTTP API 调用、Web 前端交互、Docker 一键部署，已上线可演示。

> 从 `while` 循环到 `StateGraph`，从本地玩具到线上产品，完整记录了 AI Agent 从开发到生产的过程。

📎 **在线演示**：[https://dev-agent-dovd6phmnbyxrw6qzzzyzf.streamlit.app/](https://dev-agent-dovd6phmnbyxrw6qzzzyzf.streamlit.app/)

---

## 项目结构

```
dev-agent/
├── app.py                        # Streamlit Web 入口
├── pages/
│   ├── 1_📚_知识库管理.py         # 创建 / 删除知识库
│   ├── 2_📄_文档上传.py           # 上传解析文档（PDF/Word/TXT）
│   ├── 3_💬_智能问答.py           # RAG 问答（带引用来源）
│   └── 4_📊_评估面板.py           # LLM Judge 量化评估
├── src/
│   ├── agent/
│   │   ├── dev_agent.py              # v1: 基础 Agent 循环（ReAct 模式）
│   │   ├── dev_agent_v2.py           # v2: +logging +异常保护
│   │   ├── dev_agent_v3.py           # v3: +流式输出
│   │   └── dev_agent_langgraph.py    # v4: LangGraph StateGraph
│   ├── tools/
│   │   ├── file_tools.py             # 文件工具（list / read / search）
│   │   ├── safety.py                 # 三层安全审查（黑/敏感/白名单）
│   │   ├── rag_tool.py               # 向量知识库 + RAG 检索
│   │   └── hybrid_retriever.py       # 混合检索：BM25（jieba 分词）+ 向量 + RRF
│   ├── api/
│   │   └── server.py                 # FastAPI 服务（集成 LangGraph）
│   ├── mcp_server.py                 # MCP 协议工具服务器
│   ├── config.py                     # 全局配置
│   ├── database.py                   # SQLite 数据库（知识库 + 文档管理）
│   ├── parser.py                     # 文档解析（PDF/Word/TXT/MD）
│   ├── chunker.py                    # 智能分块（500字 + 50字重叠）
│   ├── embeddings.py                 # 本地 Embedding 模型
│   ├── vector_store.py               # Chroma 向量存储
│   ├── hybrid_retriever.py           # Web 版混合检索
│   ├── rag_agent.py                  # RAG 问答 Agent
│   ├── evaluation.py                 # LLM Judge 评估（1-5 分制）
│   └── evaluate_rag.py               # RAGAS 评估脚本
├── knowledge/                        # 知识库文档
├── Dockerfile + docker-compose.yml   # Docker 部署
└── requirements.txt
```

---

## 快速开始

### 方式一：Web 前端（推荐演示用）

```bash
pip install -r requirements.txt
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### 方式二：HTTP API

```bash
pip install -r requirements.txt
python src/api/server.py
# 浏览器打开 http://localhost:8000/docs 查看 Swagger 文档
```

### 方式三：Docker 部署

```bash
docker compose up
# API: http://localhost:8000/docs
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
| **Web** | **Streamlit + SQLite + 评估面板** | **从本地 Demo 到线上产品** |

---

## Agent 工具

| 工具 | 说明 |
|------|------|
| `list_files` | 列出目录内容 |
| `read_file` | 读取文件（含三层安全审查） |
| `search_in_files` | 按关键词搜索文件 |
| `search_knowledge` | 混合检索知识库（BM25 + 向量 + RRF） |
| `add_knowledge` | 添加文本到知识库 |
| `load_file_to_knowledge` | 加载文件到知识库 |

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM | DeepSeek API |
| Agent 框架 | LangGraph（StateGraph） |
| Web 前端 | Streamlit |
| API | FastAPI + Uvicorn |
| 数据库 | SQLite |
| 向量库 | Chroma |
| 检索 | BM25（jieba 分词）+ 向量 + RRF 混合检索 |
| Embedding | text2vec-base-chinese（本地，免费） |
| 安全 | 三层审查：黑名单 + 敏感文件 + 白名单 |
| 评估 | LLM Judge（4 维度 1-5 分制）+ RAGAS |
| MCP | FastMCP |
| 部署 | Docker + Streamlit Cloud |

---

## RAG 管线

```
用户提问 → BM25（jieba 分词）→ RRF 融合 → LLM 生成答案 + 引用来源
          → 稠密向量语义检索 ↗
```

---

## 评估体系

采用 LLM-as-Judge 模式对 RAG 质量做量化评估。将「问题 + 检索结果 + 生成答案 + 参考答案」交给 DeepSeek，逐项打分（1-5 分制）：

| 指标 | 衡量什么 |
|------|------|
| Context Recall | 检索结果是否覆盖了答案所需信息 |
| Context Precision | 相关文档是否排在检索结果前面 |
| Faithfulness | 答案是否忠实于文档（有无幻觉） |
| Answer Relevancy | 答案是否直接回应了问题 |

测试集覆盖 6 种题型（定义型、对比型、推理型、应用型、刁钻型、细节型），支持自定义测试集和 A/B 对比实验。

---

## 已知问题

- Python 3.13 与 sentence-transformers 存在兼容问题，本地需 Python 3.11
- Docker 环境统一用 Python 3.11

---

## 许可证

MIT
