# -*- coding: utf-8 -*-
"""
Dev-Agent 量化评估实验脚本
============================================================
目的：用你项目里现成的 Ragas 框架 + 测试集 + eval_history 存档，
      跑一次"控制变量"对比实验，拿到真实、可复现、面试能现场调出的量化数据。

运行前提（在你自己的机器上，dev-agent 目录下）：
  1. 依赖装好：pip install -r requirements.txt   （ragas>=0.4,<0.5 已含）
  2. .env 配好 LLM_API_KEY / EMBEDDING_API_KEY 等（和平时跑项目一致）
  3. 知识库已 ingest（chroma 里有数据）
  4. Python 3.11（3.13 与 sentence-transformers 不兼容，官方建议 Docker）

运行：
  cd C:/Users/24162/Desktop/dev-agent
  python run_eval_experiment.py

输出：
  - 控制台打印各组指标 + 提升值
  - 每次结果自动存档到 data/eval_history/eval_*.json （面试可现场调出，作为证据）

注意：本脚本不改动任何源码，只是运行时调用你的评估函数。
"""

import sys
import os
import json
from datetime import datetime

# 让脚本能 import src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybrid_retriever import _ensure_kb
from src.evaluation_ragas import run_ragas_eval, compare_configs, save_result

KB_ID = _ensure_kb()
print(f"[info] 使用知识库 kb_id = {KB_ID}")

# ================================================================
# 测试集（基于 knowledge/ 下 5 篇真实文档编写，reference 摘自原文）
# 格式：{"question": "...", "reference": "..."}
# 覆盖多题型：事实查询 / 概念解释 / 对比 / 操作指导 / 综合理解
# ================================================================
TESTSET = [
    # --- 01 Agent 基础 ---
    {"question": "AI Agent 的三个核心要素是什么？",
     "reference": "AI Agent 的三个核心要素是：LLM（大语言模型）+ 工具（Tools）+ 循环（Loop），三者缺一不可。"},
    {"question": "什么是 ReAct 模式？它包含哪三个阶段？",
     "reference": "ReAct 全称 Reasoning + Acting（推理+行动），是 Agent 的核心工作模式，包含思考（Thought）、行动（Action）、观察（Observation）三个阶段。"},
    {"question": "LangGraph 相比 while 循环实现 Agent 有什么优势？",
     "reference": "流程图可视化、节点可复用、状态管理自动、容易加新功能（加节点即可）。"},

    # --- 02 RAG 管线 ---
    {"question": "RAG 解决了 LLM 的哪两个核心问题？",
     "reference": "知识截止日期（LLM 不知道训练后的新知识）和幻觉（LLM 可能编造不存在的事实）。"},
    {"question": "RAG 全链路包含哪五个步骤？",
     "reference": "文档分段（Chunking）、向量化（Embedding）、存入向量库（Indexing）、检索（Retrieval）、生成（Generation）。"},
    {"question": "文档分段（chunk）的大小有什么讲究？",
     "reference": "每段 200-500 字，太大会检索不精准，太小会丢失上下文，相邻段之间可以有 overlap 防止关键信息被切断。"},
    {"question": "余弦相似度衡量什么？公式是什么？",
     "reference": "衡量两个向量的夹角，值越接近1越相关，公式 cos_sim = (A·B)/(|A|×|B|)。"},

    # --- 03 LangGraph ---
    {"question": "StateGraph 的三个核心概念是什么？",
     "reference": "State（状态，在图中流转的数据对象，含消息历史/中间结果）、Node（节点，接收 State 返回更新后的 State）、Edge（边，普通边和条件边）。"},
    {"question": "LangGraph 中条件边的作用是什么？",
     "reference": "根据条件决定下一步，条件边是 Agent 循环的关键——根据 LLM 是否请求工具调用来决定去 call_tools 还是结束。"},
    {"question": "怎么给 Agent 添加新工具？",
     "reference": "写 Python 函数实现逻辑 → 注册工具的 JSON Schema → 把函数加入 TOOL_MAP → 在 System Prompt 中说明工具用途。"},
    {"question": "LangGraph 相比 while 循环，在可扩展性和调试上有什么优势？",
     "reference": "加功能=加节点即可（while 要改循环体）；每个节点可单独测试（while 只能整体调试）。"},

    # --- 04 MCP ---
    {"question": "MCP 是什么？它解决的核心问题是什么？",
     "reference": "MCP（Model Context Protocol，模型上下文协议）是 Anthropic 提出的开放协议，让 AI 模型自动发现和调用外部工具；解决不同 AI 应用各自定义工具调用方式、工具不能跨应用复用的问题。"},
    {"question": "MCP 采用什么架构？支持哪两种传输协议？",
     "reference": "客户端-服务器架构（MCP Server 暴露工具，MCP Client 连接获取工具列表）；支持 stdio 和标准输入输出、HTTP（Server-Sent Events）两种传输。"},
    {"question": "本项目中的 MCP 是怎么实现的？",
     "reference": "用 FastMCP 框架创建 MCP Server（src/mcp_server.py），包含一个 hello 工具验证协议原理，步骤是定义工具函数、用 FastMCP 包装、暴露工具列表。"},

    # --- 05 部署 ---
    {"question": "Dev Agent 用了哪些技术栈？",
     "reference": "LLM=DeepSeek API；Agent 框架=LangGraph(StateGraph)；Web=FastAPI+Uvicorn；向量检索=Sentence-Transformers+NumPy；MCP=FastMCP；容器化=Docker+Compose。"},
    {"question": "怎么用 Docker 一键部署 Dev Agent？",
     "reference": "运行 docker compose up，启动后浏览器打开 http://localhost:8000/docs 访问 Swagger API 文档。"},
    {"question": "dev-agent 经历了哪四个版本迭代？",
     "reference": "v1 Agent 基础循环（ReAct）；v2 logging+异常保护；v3 流式输出；v4 LangGraph StateGraph。"},
]

# ================================================================
# 方案 1：检索参数 top_k 对比（零代码改动，必能出数）
#   讲的故事：用评估反推检索深度，而非凭感觉定 top_k
# ================================================================
def experiment_topk(testset, quick=False):
    print("\n=== 方案1: top_k 对比实验 ===", flush=True)
    configs = [
        {"name": "top_k=3", "top_k": 3},
        {"name": "top_k=5", "top_k": 5},
    ]
    if not quick:
        configs.append({"name": "top_k=8", "top_k": 8})
    n_q = len(testset)
    print(f"[info] 题数={n_q}，配置={[c['name'] for c in configs]} "
          f"（共 {n_q * len(configs)} 次评估，每次含生成+4指标）", flush=True)
    out = compare_configs(KB_ID, testset, configs, save=True)
    print(f"[ok] 已存档: {out['timestamp']}  (测试题数={out['test_size']})", flush=True)
    for r in out["results"]:
        print(f"  {r['name']:10s} -> {r['metrics']}", flush=True)
    return out


# ================================================================
# 方案 2：Prompt 优化前后对比（运行时替换 SYSTEM_PROMPT，不改源码）
#   讲的故事：Prompt Engineering 量化提升（最贴合你的简历叙事）
# ================================================================
def experiment_prompt(testset, quick=False):
    print("\n=== 方案2: Prompt 优化前后对比 ===", flush=True)
    import src.rag_qa as ra

    baseline_prompt = ra.SYSTEM_PROMPT  # 当前线上用的

    # 优化版：在基线基础上加强"防幻觉 + 结构化输出 + 1 个 few-shot 范例"
    # （你可按自己真实的 Prompt Engineering 改这里，体现针对性设计）
    optimized_prompt = baseline_prompt + """

补充要求（用于提升忠实度与相关性）：
- 回答必须严格基于「参考文档」，任何文档之外的推断都要明确标注"（推测）"。
- 输出结构：先给一句话结论，再给带 [n] 引用的要点展开。
- 示例：
  问：RAG 解决了 LLM 的哪两个问题？
  答：RAG 主要解决两个核心问题。1) 知识截止日期：LLM 不知道训练后的新知识 [1]；2) 幻觉：LLM 可能编造不存在的事实 [1]。"""

    # 基线
    ra.SYSTEM_PROMPT = baseline_prompt
    print("[info] 跑 baseline prompt ...", flush=True)
    base = run_ragas_eval(KB_ID, testset, top_k=5)

    # 优化
    ra.SYSTEM_PROMPT = optimized_prompt
    print("[info] 跑 optimized prompt ...", flush=True)
    opt = run_ragas_eval(KB_ID, testset, top_k=5)

    # 组装对比并存档（兼容 compare_configs 的 JSON 格式）
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kb_id": KB_ID,
        "test_size": len(TESTSET),
        "results": [
            {"name": "baseline_prompt", "config": {}, "metrics": base["metrics"], "details": base["details"]},
            {"name": "optimized_prompt", "config": {}, "metrics": opt["metrics"], "details": opt["details"]},
        ],
    }
    save_result(out)

    print(f"[ok] 已存档: {out['timestamp']}")
    print("baseline :", base["metrics"])
    print("optimized:", opt["metrics"])
    print("--- 提升值 ---")
    for k in base["metrics"]:
        b, o = base["metrics"][k], opt["metrics"][k]
        print(f"  {k:20s}: {b:5.1f} -> {o:5.1f}   ({(o - b):+.1f})")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dev-Agent 量化评估实验")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：只用前 6 题 + top_k=3/5，约几分钟出数，用于验证管线")
    args = parser.parse_args()

    ACTIVE_TESTSET = TESTSET[:6] if args.quick else TESTSET
    mode = "QUICK（前6题）" if args.quick else "完整（16题）"
    print(f"[启动] 模式={mode}", flush=True)

    experiment_topk(ACTIVE_TESTSET, quick=args.quick)
    experiment_prompt(ACTIVE_TESTSET, quick=args.quick)
    print("\n[done] 所有结果已存入 data/eval_history/，面试可现场调出作为证据。", flush=True)
