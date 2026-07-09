# Dev Agent 部署指南

## 项目概述

Dev Agent 是一个基于 LangGraph + FastAPI 的 AI 开发助手，能操作本地文件、搜索知识库。支持 HTTP API 调用和 Docker 一键部署。

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| LLM | DeepSeek API |
| Agent 框架 | LangGraph (StateGraph) |
| Web 框架 | FastAPI + Uvicorn |
| 向量检索 | Sentence-Transformers + NumPy（自实现） |
| MCP 协议 | FastMCP |
| 容器化 | Docker + Docker Compose |

## 部署方式

### Docker 一键部署（推荐）

只需一条命令即可启动整个服务：

```bash
docker compose up
```

服务启动后，浏览器打开 http://localhost:8000/docs 即可访问 Swagger API 文档。

Docker 环境统一使用 Python 3.11，避免了 Python 3.13 与 sentence-transformers 的兼容问题。docker-compose.yml 配置了端口映射（8000:8000）和桌面目录挂载，让容器内的 Agent 可以访问宿主机的文件。

### 本地运行

如果不想用 Docker，也可以本地运行：

```bash
pip install -r requirements.txt
python src/api/server.py
```

但需要注意：Python 3.13 与 sentence-transformers 存在兼容问题，本地需要 Python 3.11 才能正常使用 RAG 功能。这也是推荐使用 Docker 的原因——环境一致性。

## API 使用

发送聊天请求：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "列出桌面的文件", "work_dir": "/app/host-desktop"}'
```

API 支持流式输出（Server-Sent Events），前端可以实时看到 Agent 的思考过程。

## 项目版本演进

dev-agent 经历了四个版本的迭代：

| 版本 | 核心改进 | 解决的问题 |
|------|---------|-----------|
| v1 | Agent 基础循环 | 实现 ReAct 模式：思考 → 调工具 → 回答 |
| v2 | logging + 异常保护 | 单个工具崩溃不影响整个 Agent |
| v3 | 流式输出 | 打字机效果，用户体验更好 |
| v4 | LangGraph StateGraph | 流程可视图化，架构更清晰 |

这个演进过程展示了从一个简单的 while 循环到完整的图状态机的架构升级路径，每一步都解决了实际开发中遇到的具体问题。

## Dockerfile 和 docker-compose 的作用

Dockerfile 定义了如何构建项目镜像：基于 Python 3.11，安装依赖，复制源码，指定启动命令。docker-compose.yml 定义了如何运行服务：端口映射、环境变量、卷挂载（把宿主机目录映射到容器内）。两者的关系是：Dockerfile 负责"构建什么"，docker-compose.yml 负责"怎么运行"。
