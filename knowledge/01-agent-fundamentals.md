# Agent 基础原理

## AI Agent 的核心三要素

AI Agent 的三个核心要素是：**LLM（大语言模型）+ 工具（Tools）+ 循环（Loop）**。

- **LLM**：作为 Agent 的大脑，负责理解用户意图、决定下一步行动、生成最终回答。本项目使用 DeepSeek API。
- **工具**：让 Agent 能执行实际操作，比如读写文件、搜索知识库。每个工具都有 JSON Schema 描述，LLM 根据 Schema 决定调用哪个工具。
- **循环**：Agent 不是一次性回答，而是"思考 → 行动 → 观察 → 再思考"的循环过程，直到问题解决。

三者缺一不可：没有 LLM，Agent 无法理解自然语言；没有工具，Agent 只能聊天不能做事；没有循环，Agent 无法处理复杂任务。

## ReAct 模式

ReAct 全称 Reasoning + Acting（推理 + 行动），是 Agent 的核心工作模式。

ReAct 循环包含三个阶段：
1. **思考（Reasoning/Thought）**：分析用户问题，判断需要什么信息，决定下一步做什么
2. **行动（Action）**：调用工具执行具体操作，比如读文件、搜索知识库
3. **观察（Observation）**：接收工具返回的结果，判断是否足够回答问题，如果不够就继续循环

这个循环一直持续，直到 LLM 认为信息足够，生成最终回答。这种模式让 Agent 能处理单次推理无法解决的复杂问题。

## 如何在代码中实现 Agent 循环

最简单的实现是用 `while` 循环：LLM 不断决策，直到不再需要调用工具。本项目 v1 版本就是这样实现的。

更优雅的方式是用 LangGraph 的 StateGraph：把 Agent 流程定义成节点（node）和边（edge）组成的图。节点包括 call_model（调用 LLM）和 call_tools（调用工具），条件边根据 LLM 是否要求调用工具来决定下一步。

LangGraph 相比 while 循环的优势在于：流程图可视化、节点可复用、状态管理自动、容易加新功能（加节点即可）。
