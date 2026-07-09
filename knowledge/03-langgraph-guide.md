# LangGraph 实战指南

## LangGraph 是什么

LangGraph 是 LangChain 团队开发的一个 Agent 编排框架，用**图（Graph）**的概念来组织 Agent 的工作流程。它的核心是 StateGraph：把 Agent 流程定义成节点和边的有向图，状态在节点之间流转。

## StateGraph 的核心概念

StateGraph 有三个核心概念：

1. **State（状态）**：在整个图中流转的数据对象，通常包含消息历史、中间结果等。在 dev-agent 项目中，State 包含 messages（对话历史）。

2. **Node（节点）**：图中的处理单元，接收当前 State，返回更新后的 State。本项目有两个核心节点：
   - `call_model`：调用 LLM，让 LLM 决定下一步
   - `call_tools`：执行 LLM 请求的工具调用

3. **Edge（边）**：节点之间的连接。分为普通边（无条件流转）和条件边（根据条件决定下一步）。条件边是 Agent 循环的关键——根据 LLM 是否请求工具调用来决定是去 call_tools 还是结束。

## Agent 执行流程

用户提问 → call_model（LLM 决策）→ 条件判断：
- 如果需要调工具 → call_tools（执行工具）→ 回到 call_model
- 如果不需要 → 结束，返回最终回答

这个循环会一直持续，直到 LLM 认为信息足够回答用户问题。

## LangGraph 和 while 循环的对比

| 维度 | LangGraph StateGraph | while 循环 |
|------|---------------------|-----------|
| 流程可视化 | ✅ 图结构清晰可见 | ❌ 流程隐藏在代码中 |
| 可扩展性 | ✅ 加功能 = 加节点 | ❌ 改逻辑要改循环体 |
| 状态管理 | ✅ 自动管理 State 流转 | ❌ 手动维护变量 |
| 调试 | ✅ 每个节点可单独测试 | ❌ 只能整体调试 |
| 学习曲线 | 稍高 | 低 |
| 适用场景 | 复杂多步骤流程 | 简单线性流程 |

LangGraph 的核心优势是：流程是显式的（图），不是隐式的（代码逻辑）。这让你能清楚地看到 Agent 的工作流程，也方便后续扩展。

## 怎么给 Agent 添加新工具

添加新工具的步骤：
1. 写一个 Python 函数，实现工具的具体逻辑
2. 注册工具的 JSON Schema（名称、描述、参数定义），让 LLM 知道这个工具的存在
3. 把函数加入 TOOL_MAP（工具名 → 函数映射）
4. 在 System Prompt 中说明工具的用途

LLM 会根据 JSON Schema 自动决定何时调用这个工具，不需要额外编码。
