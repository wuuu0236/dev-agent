# legacy / 历史版本

这里保留 Agent 的三个历史架构版本，用于展示演进过程。**当前生产版本是 v4**，见 `../dev_agent_langgraph.py`。

| 版本 | 文件 | 改进点 | 解决的问题 |
|:---:|------|------|------|
| v1 | dev_agent.py | Agent 基础循环 | 思考 -> 调工具 -> 回答的最小可用版 |
| v2 | dev_agent_v2.py | logging + 异常保护 | 工具崩溃不连累 Agent 主循环 |
| v3 | dev_agent_v3.py | 流式输出 | 打字机效果，用户不再干等 |
| v4 | ../dev_agent_langgraph.py | LangGraph StateGraph | 流程可视化，加功能只需加节点 |

## 运行方式

从**项目根目录**运行（不是从本目录）：

```bash
python -m src.agent.legacy.dev_agent        # v1
python -m src.agent.legacy.dev_agent_v2     # v2
python -m src.agent.legacy.dev_agent_v3     # v3
```

## 注意

这三个版本引用的是**早期检索实现**（`src/tools/hybrid_retriever.py`、`src/tools/rag_tool.py`），与当前线上 demo 使用的 `src/hybrid_retriever.py` 不是同一套。它们只为展示 **Agent 循环本身**的演进，不代表当前检索方案。
