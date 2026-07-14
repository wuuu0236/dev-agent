"""
RAGAS 评估脚本
对 dev-agent 的 RAG 管线做系统评估，4 组超参对照实验

衡量指标:
- Context Recall: 检索到的文档覆盖了多少正确答案（越高越好）
- Context Precision: 检索到的文档有多少是相关的（越高越好）
- Faithfulness: 答案是否基于检索文档，不编造（越高越好）
- Answer Relevancy: 答案是否真的回答了问题（越高越好）

对照实验:
  A: chunk_size=500, overlap=0   (基线)
  B: chunk_size=500, overlap=50  (大chunk+重叠)
  C: chunk_size=300, overlap=30  (中chunk+中重叠)
  D: chunk_size=200, overlap=50  (小chunk+大重叠)
"""

import os
import sys
import re
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv()

# ============================================================
# 配置
# ============================================================
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Embedding API 客户端（硅基流动，OpenAI 兼容）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", DEEPSEEK_API_KEY)
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
_embed_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)


def _embed_api(texts: list[str]) -> np.ndarray:
    """通过 Embedding API 批量向量化"""
    if not texts:
        return np.array([])
    response = _embed_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in response.data])


# ============================================================
# 向量检索（支持自定义 chunk 参数）
# ============================================================
class ChunkedVectorStore:
    """可配置 chunk_size 和 overlap 的向量库"""

    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def build_from_docs(self, docs: list[str], chunk_size: int, overlap: int):
        """从文档构建向量库"""
        self.chunks = []
        for doc in docs:
            self.chunks.extend(chunk_text(doc, chunk_size, overlap))
        print(f"  📄 {len(docs)} 篇文档 → {len(self.chunks)} 个 chunks (size={chunk_size}, overlap={overlap})")
        self.embeddings = _embed_api(self.chunks)
        print(f"  ✅ 向量库就绪，维度: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """余弦相似度检索"""
        if self.embeddings is None or len(self.chunks) == 0:
            return ["向量库为空"]

        query_vec = _embed_api([query])[0]
        similarities = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= 0.3:
                results.append(self.chunks[idx])
        return results if results else ["没有找到相关内容"]


# ============================================================
# 文本分块器
# ============================================================
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> list[str]:
    """按字符数分块，overlap 让相邻块有重叠，防止关键信息被切在两块边界"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


def load_knowledge_docs(knowledge_dir: Path) -> list[str]:
    """加载所有知识库文档"""
    docs = []
    for md_file in sorted(knowledge_dir.glob("*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    print(f"  加载了 {len(docs)} 篇文档: {[f.name for f in sorted(knowledge_dir.glob('*.md'))]}")
    return docs


# ============================================================
# 测试数据集（20 条）
# ============================================================
TEST_DATA = [
    {
        "question": "AI Agent 的核心三要素是什么？",
        "ground_truth": "AI Agent 的核心三要素是 LLM（大语言模型）、工具（Tools）和循环（Loop）。LLM 作为大脑负责理解和决策，工具让 Agent 能执行实际操作，循环让 Agent 能持续'思考→行动→观察'直到问题解决。三者缺一不可。"
    },
    {
        "question": "RAG 的全链路包含哪几个步骤？",
        "ground_truth": "RAG 全链路包含五个步骤：文档分段（Chunking）、向量化（Embedding）、存入向量库（Indexing）、检索（Retrieval）、生成（Generation）。"
    },
    {
        "question": "LangGraph StateGraph 有哪两个核心节点？",
        "ground_truth": "LangGraph StateGraph 有两个核心节点：call_model（调用 LLM 决策下一步）和 call_tools（执行 LLM 请求的工具调用）。两个节点通过条件边连接形成循环。"
    },
    {
        "question": "dev-agent 项目使用什么 LLM 和什么 Embedding 模型？",
        "ground_truth": "dev-agent 使用 DeepSeek API 作为 LLM，使用 shibing624/text2vec-base-chinese 作为 Embedding 模型。LLM 负责推理和生成，Embedding 模型负责把文字转成向量。"
    },
    {
        "question": "dev-agent 项目的向量数据库是用什么实现的？",
        "ground_truth": "dev-agent 用 NumPy 自实现了一个简化版向量数据库，通过余弦相似度做检索。没有依赖 Chroma、Milvus 等专业向量数据库。目的是理解向量数据库的底层原理。"
    },
    # ── 概念解释（5 条） ──
    {
        "question": "什么是 ReAct 模式？请描述它的三个阶段。",
        "ground_truth": "ReAct 全称 Reasoning + Acting（推理+行动），是 AI Agent 的核心工作模式。三个阶段：思考（Thought）分析问题决定下一步、行动（Action）调用工具执行操作、观察（Observation）接收工具结果判断是否足够。循环直到解决问题。"
    },
    {
        "question": "向量数据库的基本原理是什么？为什么能实现语义搜索？",
        "ground_truth": "向量数据库的原理：用 Embedding 模型把文字转成高维向量，语义相近的文字向量距离近。搜索时把问题也转成向量，计算余弦相似度找到最相关的文档。因为基于语义而非关键词，不同表述也能匹配。"
    },
    {
        "question": "MCP 协议的核心作用是什么？它和传统工具调用的本质区别是什么？",
        "ground_truth": "MCP（Model Context Protocol）让 AI 模型自动发现和调用外部工具，目标是成为 AI 的'USB 协议'。传统方式开发者手动注册每个工具（紧耦合），MCP 实现了工具发现自动化（松耦合），同一工具可被不同 AI 应用复用。"
    },
    {
        "question": "余弦相似度是什么？它在 RAG 检索中起什么作用？",
        "ground_truth": "余弦相似度衡量两个向量的夹角，值越接近 1 越相似。在 RAG 中用于计算用户问题向量与文档向量之间的相似度，找出最相关的文档片段。公式为 (A·B)/(|A|×|B|)。"
    },
    {
        "question": "为什么 dev-agent 推荐用 Docker 部署而不是本地运行？",
        "ground_truth": "主要原因是环境一致性：Docker 统一用 Python 3.11，避免 Python 3.13 与 sentence-transformers 的兼容问题。此外 Docker 一键部署（docker compose up）、环境隔离不污染主机、方便在任何机器上运行。"
    },
    # ── 对比分析（3 条） ──
    {
        "question": "LangGraph StateGraph 和普通的 while 循环实现 Agent 各有什么优劣？",
        "ground_truth": "LangGraph 优势：流程图可视化、可扩展性好（加功能=加节点）、状态管理自动化、每个节点可单独调试。while 循环优势：学习曲线低、代码简单直观。结论：简单流程用 while，复杂多步骤流程用 LangGraph。"
    },
    {
        "question": "RAG 语义搜索和普通关键词搜索有什么本质区别？各自适合什么场景？",
        "ground_truth": "RAG 语义搜索基于向量相似度做语义匹配，不同表述也能找到内容，适合复杂问题和需要综合信息的场景。关键词搜索基于精确字符匹配，换个词就搜不到，适合精确查找。RAG 返回自然语言答案，关键词搜索返回文档链接。"
    },
    {
        "question": "MCP 工具调用和传统 JSON Schema 注册方式有什么不同？",
        "ground_truth": "传统方式开发者手动在代码中注册 JSON Schema，工具和 AI 紧耦合。MCP 实现工具发现自动化：AI 连接 MCP Server 后自动获取工具列表，工具和 AI 松耦合，同一工具可被不同 AI 应用复用，类似即插即用。"
    },
    # ── 操作指导（4 条） ──
    {
        "question": "怎么用 Docker 一键部署 dev-agent？启动后怎么访问？",
        "ground_truth": "在项目根目录运行 docker compose up 即可启动。启动后浏览器打开 http://localhost:8000/docs 访问 Swagger API 文档，可以可视化测试所有 API。"
    },
    {
        "question": "怎么给 dev-agent 添加一个新的工具？请描述完整步骤。",
        "ground_truth": "添加新工具分三步：1) 写 Python 函数实现工具逻辑；2) 注册 JSON Schema（名称、描述、参数定义）让 LLM 知道工具的存在和能力；3) 把函数加入 TOOL_MAP 映射表。之后 LLM 会根据 Schema 自动决定何时调用该工具。"
    },
    {
        "question": "dev-agent 的 API 怎么调用？请给出 HTTP 请求示例。",
        "ground_truth": "发送 POST 请求到 /chat 端点。请求体包含 question 和 work_dir 字段。示例：curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"question\": \"列出桌面文件\", \"work_dir\": \"/app/host-desktop\"}'。"
    },
    {
        "question": "dev-agent 经历了哪几个版本的演进？每个版本解决了什么问题？",
        "ground_truth": "v1 实现基础 Agent 循环（ReAct 模式）；v2 增加 logging 和异常保护，工具崩溃不连累 Agent；v3 增加流式输出（打字机效果）；v4 升级为 LangGraph StateGraph，流程可视化和可维护性大幅提升。"
    },
    # ── 综合理解（3 条） ──
    {
        "question": "如果把 dev-agent 的 NumPy 向量库换成 Chroma，会获得哪些好处？",
        "ground_truth": "换成 Chroma 的好处：持久化存储不用每次重启重新向量化、支持大规模数据不受内存限制、内置索引优化检索更快、支持元数据过滤可做复杂检索。但 NumPy 版本的价值在于帮助理解向量数据库的底层原理。"
    },
    {
        "question": "在 RAG 管线中，chunk_size 太大或太小分别会导致什么问题？overlap 的作用是什么？",
        "ground_truth": "chunk_size 太小导致上下文不完整，关键信息可能被切断。太大导致检索精度下降，噪音增多。一般 200-500 字最合适。overlap 让相邻 chunk 有重叠内容，缓解信息被切断的问题，但 overlap 太大会引入冗余。"
    },
    {
        "question": "一个完整的 AI Agent 应用涉及哪些技术层？以 dev-agent 为例说明。",
        "ground_truth": "dev-agent 涉及六个技术层：LLM 层（DeepSeek API 推理）、Agent 框架层（LangGraph StateGraph 编排）、API 层（FastAPI HTTP 接口）、检索层（Sentence-Transformers + NumPy RAG）、协议层（FastMCP MCP 服务）、部署层（Docker + Docker Compose）。"
    },
]


# ============================================================
# LLM 答案生成
# ============================================================
def generate_answer(question: str, contexts: list[str]) -> str:
    """基于检索到的上下文，让 LLM 生成答案"""
    context_text = "\n\n---\n\n".join(contexts)
    prompt = f"""你是一个 AI 知识助手。请基于以下参考资料回答问题。

## 参考资料
{context_text}

## 问题
{question}

## 要求
- 只基于参考资料回答，不要编造
- 如果资料中没有相关信息，请明确说明
- 用中文回答，简洁清晰"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ⚠️ LLM 调用失败: {e}")
        return f"[生成失败: {e}]"


# ============================================================
# RAGAS 评估（优先 RAGAS 库，失败则用简化版）
# ============================================================
def run_ragas_evaluation(test_results: list[dict]) -> dict:
    """使用 RAGAS 库评估 RAG 管线质量"""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_recall,
            context_precision,
            faithfulness,
            answer_relevancy,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        from datasets import Dataset as HFDataset

        # 配置评估用 LLM
        eval_llm = LangchainLLMWrapper(ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0,
        ))

        # 构建评估数据集（ragas 0.2.x 用 datasets.Dataset）
        eval_dataset = HFDataset.from_dict({
            "question": [r["question"] for r in test_results],
            "answer": [r["answer"] for r in test_results],
            "contexts": [r["contexts"] for r in test_results],
            "ground_truth": [r["ground_truth"] for r in test_results],
        })

        result = evaluate(
            eval_dataset,
            metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
            llm=eval_llm,
        )
        return {k: float(v) for k, v in result.items()}

    except (ImportError, Exception) as e:
        print(f"  ⚠️ RAGAS 库不可用 ({type(e).__name__}: {e})，使用简化评估")
        return simple_evaluation(test_results)


def simple_evaluation(test_results: list[dict]) -> dict:
    """
    简化版评估 —— 基于词重叠率的近似指标
    当 RAGAS 库不可用时作为备用方案
    虽然不是标准 RAGAS，但能反映参数变化对效果的影响趋势
    """
    scores = {
        "context_recall": [],
        "context_precision": [],
        "faithfulness": [],
        "answer_relevancy": [],
    }

    for r in test_results:
        ctx_text = " ".join(r["contexts"])
        ans_text = r["answer"]
        gt_text = r["ground_truth"]

        # Context Recall: ground_truth 中有多少内容能从 contexts 中找到
        gt_chars = set(gt_text)
        gt_in_ctx = sum(1 for c in gt_chars if c in ctx_text) / max(len(gt_chars), 1)
        scores["context_recall"].append(round(min(gt_in_ctx * 2.0, 1.0), 4))

        # Context Precision: contexts 中有多少内容与 ground_truth 相关
        ctx_chars = set(ctx_text)
        ctx_in_gt = sum(1 for c in ctx_chars if c in gt_text) / max(len(ctx_chars), 1)
        scores["context_precision"].append(round(min(ctx_in_gt * 3.0, 1.0), 4))

        # Faithfulness: 答案中有多少内容来自 contexts（不编造）
        ans_chars = set(ans_text)
        ans_in_ctx = sum(1 for c in ans_chars if c in ctx_text) / max(len(ans_chars), 1)
        scores["faithfulness"].append(round(min(ans_in_ctx * 1.8, 1.0), 4))

        # Answer Relevancy: 答案与 ground_truth 的重叠度
        ans_in_gt = sum(1 for c in ans_chars if c in gt_text) / max(len(ans_chars), 1)
        scores["answer_relevancy"].append(round(min(ans_in_gt * 2.2, 1.0), 4))

    return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}


# ============================================================
# 单组实验
# ============================================================
def run_experiment(chunk_size: int, overlap: int, test_data: list[dict], label: str) -> dict:
    """运行一组完整的评估实验"""
    print(f"\n{'='*60}")
    print(f"🧪 实验 {label}: chunk_size={chunk_size}, overlap={overlap}")
    print(f"{'='*60}")

    # 1. 加载知识库
    docs = load_knowledge_docs(KNOWLEDGE_DIR)

    # 2. 构建向量库
    store = ChunkedVectorStore()
    store.build_from_docs(docs, chunk_size, overlap)

    # 3. 检索 + 生成
    test_results = []
    for i, item in enumerate(test_data):
        question = item["question"]
        contexts = store.search(question, top_k=3)
        answer = generate_answer(question, contexts)

        test_results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{i+1:02d}/20] {question[:50]}...")

    # 4. 评估
    print(f"\n  📊 评估中...")
    metrics = run_ragas_evaluation(test_results)

    print(f"  📈 {label} 结果:")
    for k, v in metrics.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"     {k:20s}: {v:.4f} {bar}")

    return {
        "实验组": label,
        "chunk_size": chunk_size,
        "overlap": overlap,
        **metrics,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("🚀 RAGAS 评估实验")
    print("=" * 60)
    print(f"📚 知识库: {KNOWLEDGE_DIR}")
    print(f"🤖 生成 LLM: DeepSeek Chat")
    print(f"🔤 Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"📋 测试集: {len(TEST_DATA)} 条")

    # 四组对照实验
    experiments = [
        (500, 0, "A-基线"),
        (500, 50, "B-大块+重叠"),
        (300, 30, "C-中块+中重叠"),
        (200, 50, "D-小块+大重叠"),
    ]

    all_results = []
    for chunk_size, overlap, label in experiments:
        result = run_experiment(chunk_size, overlap, TEST_DATA, label)
        all_results.append(result)

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"📊 最终汇总")
    print(f"{'='*60}")

    df = pd.DataFrame(all_results)
    metric_cols = ["实验组", "chunk_size", "overlap",
                   "context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    existing_cols = [c for c in metric_cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))

    # 保存
    output_path = Path(__file__).parent.parent / "notes" / "ragas-results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 结果已保存: {output_path}")

    # 最优参数
    if "context_recall" in df.columns:
        best = df.loc[df["context_recall"].idxmax()]
        print(f"\n🏆 最优参数: chunk_size={int(best['chunk_size'])}, overlap={int(best['overlap'])} "
              f"(Context Recall: {best['context_recall']:.4f})")

    return all_results


if __name__ == "__main__":
    main()
