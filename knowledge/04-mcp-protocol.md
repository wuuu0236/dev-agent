# MCP 协议详解

## MCP 是什么

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 提出的一个开放协议，目的是让 AI 模型能够自动发现和调用外部工具。它的核心理念是：让 AI 自己去发现有什么工具可用，而不是开发者手动告诉 AI 每个工具的参数。

MCP 解决的核心问题是：不同 AI 应用各自定义自己的工具调用方式，导致工具不能跨应用复用。MCP 提供了一个统一的标准协议。

## MCP 的工作原理

MCP 采用客户端-服务器架构：

1. **MCP Server**：工具提供方。暴露自己提供的工具列表（名称、描述、参数 Schema）。当客户端调用时，执行工具并返回结果。

2. **MCP Client**：AI 应用端（如 Claude Desktop）。连接到 MCP Server，获取工具列表，让 AI 模型自主选择调用哪个工具。

通信方式支持两种传输协议：标准输入输出（stdio）和 HTTP（Server-Sent Events）。

## 本项目中的 MCP 实现

本项目使用 FastMCP 框架创建了一个 MCP Server（`src/mcp_server.py`），包含一个 hello 工具用于验证 MCP 协议的工作原理。关键代码只需几步：定义工具函数、用 FastMCP 包装、暴露工具列表。

## MCP 与其他工具调用的区别

传统方式中，开发者需要在代码中手动注册每个工具的 Schema，工具和 AI 应用是紧耦合的。MCP 则实现了工具发现自动化——AI 连接到 MCP Server 后自动获取可用工具列表，工具和 AI 应用是松耦合的，同一个工具可以被不同 AI 应用复用。

## MCP 的生态定位

MCP 的目标是成为 AI 应用的"USB 协议"：就像 USB 让任何外设都能即插即用，MCP 让任何工具都能被任何 AI 应用发现和调用。目前 Claude Desktop 原生支持 MCP，越来越多的 AI 工具平台也在接入这个协议。
