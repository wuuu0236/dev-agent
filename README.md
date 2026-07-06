# Dev Agent —— 从零构建的开发助手 Agent

## 项目结构
```
dev-agent/
├── src/
│   ├── agent/
│   │   ├── dev_agent.py       # v1: 基础 Agent 循环
│   │   ├── dev_agent_v2.py    # v2: +logging +异常保护
│   │   └── dev_agent_v3.py    # v3: +流式输出
│   └── tools/
│       ├── file_tools.py       # 文件系统工具（列目录、读文件、搜索）
│       └── rag_tool.py         # RAG 工具（向量库、语义搜索、文档入库）
├── tests/
└── notes/
    └── maxkb-vs-handwrite.md   # MaxKB 与手写 RAG 对比
```

## 我学会了什么

### 1. Agent 的核心三要素
- **LLM（大脑）**：调 DeepSeek API，不训练
- **工具（手脚）**：普通 Python 函数 + JSON Schema 描述
- **循环（决策链）**：while 循环，每轮决定「继续调工具」还是「回答」

### 2. 工具调用全链路
```
用户提问 → messages 发给 AI → AI 返回 tool_calls
→ 执行 Python 函数 → 结果塞回 messages
→ 再发给 AI → 循环直到 AI 决定回答
```

### 3. RAG = 向量化 + 检索
- 向量就是坐标，语义相近的词坐标接近
- Embedding 模型把文字转成坐标
- 余弦相似度计算距离，越近越相关

### 4. 企业级改进
- print() → logging（可控制级别、可追溯）
- 工具执行加 try-except（单个工具挂了不影响全局）
- 流式输出（stream=True，打字机效果）

## MaxKB 对照
MaxKB 的每个功能我都能在自己代码里找到对应实现。
它是成熟产品，我写的是教学代码，但原理相同。

## 技术栈
- Python · OpenAI SDK · DeepSeek API
- Sentence-Transformers · NumPy
- logging · atexit
