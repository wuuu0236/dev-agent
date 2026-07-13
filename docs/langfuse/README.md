截图清单（跑通 Langfuse 后截取，命名如下）：

1. 01-trace-list.png — Traces 列表页，能看到 dev-agent-langgraph 与 rag.query
2. 02-trace-detail-agent.png — 展开一条 Agent trace，节点时序图 call_model → call_tools → call_model
3. 03-trace-detail-rag.png — 展开一条 RAG trace，rag.query → rag.generate_answer + openai span
