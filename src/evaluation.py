"""
自定义 RAG 评估模块（不依赖 RAGAS，避免兼容性问题）

用 DeepSeek 做 LLM Judge，直接评判检索和生成质量：
  - Recall：检索到的文档是否包含参考答案的关键信息
  - Precision：相关文档是否排在前面
  - Faithfulness：答案是否完全来自检索文档（有无幻觉）
  - Relevancy：答案是否与问题相关

原理：把评估本身也当成一个 LLM 任务——
  给 LLM 看「问题 + 检索结果 + 生成的答案 + 参考答案」，
  让 LLM 逐项打分（1-5），返回结构化结果。
"""
import json
from openai import OpenAI
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.rag_agent import rag_query
from src.hybrid_retriever import HybridRetriever

_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

JUDGE_PROMPT = """你是一个 RAG 系统评估专家。请根据以下信息，对 RAG 系统的表现打分。

## 用户问题
{question}

## 参考答案（人工编写）
{reference}

## 检索到的文档片段
{contexts}

## 系统生成的答案
{answer}

## 评分要求
对以下四个维度分别打分（1-5 分，5 分最高）：

1. **Context Recall（召回率）**：检索到的文档片段中，是否包含参考答案的关键信息？
   - 5: 全部关键信息都在检索结果中
   - 3: 部分关键信息在检索结果中
   - 1: 检索结果完全不相关

2. **Context Precision（精确率）**：检索到的文档中，相关的文档是否排在前面？
   - 5: 最相关的文档排在最前面
   - 3: 相关文档散落在各处
   - 1: 相关文档排在最后

3. **Faithfulness（忠实度）**：生成的答案是否完全基于检索到的文档？（有没有编造文档中没有的信息）
   - 5: 完全忠实于文档，没有编造
   - 3: 大部分忠实，有少量无关信息
   - 1: 大量编造，脱离文档

4. **Answer Relevancy（答案相关性）**：答案是否直接回应了用户的问题？
   - 5: 完全切题，直接回答问题
   - 3: 部分相关，有些偏离
   - 1: 答非所问

## 输出格式（严格 JSON）
```json
{{
  "context_recall": 4,
  "context_precision": 5,
  "faithfulness": 4,
  "answer_relevancy": 5,
  "comment": "简短评价（一句话）"
}}
```"""


def judge_single(question: str, reference: str, contexts: list[str], answer: str) -> dict:
    """用 LLM 对单个问题的 RAG 表现打分"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        contexts="\n\n---\n\n".join(contexts[:5]),
        answer=answer
    )

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )

    raw = response.choices[0].message.content

    # 提取 JSON（容错：LLM 可能在 JSON 外面包了 markdown 标记）
    try:
        # 尝试直接解析
        return json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 ```json ... ``` 中的内容
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "JSON 解析失败", "raw": raw[:200]}


def run_evaluation(kb_id: str, test_questions: list[dict]) -> dict:
    """
    运行自定义评估（LLM Judge 模式）。

    test_questions: [{"question": "...", "reference": "..."}, ...]

    返回：{metrics: {avg_recall, avg_precision, avg_faithfulness, avg_relevancy}, details: [...]}
    """
    if not test_questions:
        return {"error": "测试集为空"}

    all_scores = {"context_recall": [], "context_precision": [], "faithfulness": [], "answer_relevancy": []}
    details = []

    for item in test_questions:
        # 跑 RAG
        result = rag_query(kb_id, item["question"])
        retriever = HybridRetriever(kb_id)
        contexts = retriever.search(item["question"], top_k=5)
        context_texts = [c["content"] for c in contexts]

        # LLM 打分
        scores = judge_single(
            question=item["question"],
            reference=item["reference"],
            contexts=context_texts,
            answer=result["answer"]
        )

        if "error" not in scores:
            all_scores["context_recall"].append(scores["context_recall"])
            all_scores["context_precision"].append(scores["context_precision"])
            all_scores["faithfulness"].append(scores["faithfulness"])
            all_scores["answer_relevancy"].append(scores["answer_relevancy"])

        details.append({
            "question": item["question"],
            "answer": result["answer"],
            "scores": scores
        })

    # 计算平均分
    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    metrics = {
        "context_recall": avg(all_scores["context_recall"]),
        "context_precision": avg(all_scores["context_precision"]),
        "faithfulness": avg(all_scores["faithfulness"]),
        "answer_relevancy": avg(all_scores["answer_relevancy"])
    }

    return {"metrics": metrics, "details": details}


def get_demo_questions() -> list[dict]:
    """内置演示测试集"""
    return [
        {
            "question": "LangGraph 是什么？它和 LangChain 有什么关系？",
            "reference": "LangGraph 是 LangChain 生态系统中的一个库，用于构建有状态的、多参与者的应用程序。它扩展了 LangChain，通过图结构来编排 Agent 的执行流程。"
        },
        {
            "question": "在 LangGraph 中，StateGraph 的作用是什么？",
            "reference": "StateGraph 是 LangGraph 的核心类，用于定义 Agent 的状态图。它管理状态在各节点之间的流转，支持条件边来决定下一步执行哪个节点。"
        },
        {
            "question": "为什么 RAG 系统需要文档分块？",
            "reference": "因为 LLM 上下文窗口有限，不能一次处理整个文档；大块文本包含无关信息影响检索精度。分块后可以更精确地定位相关段落。"
        },
        {
            "question": "向量检索和 BM25 各自的优缺点？",
            "reference": "向量检索擅长语义匹配但可能漏掉精确关键词；BM25 擅长关键词匹配但不理解语义。两者结合互补。"
        },
        {
            "question": "什么是 RRF？",
            "reference": "RRF 是一种融合多个搜索结果排名的方法，基于每个文档在各系统中的排名来计算最终得分，避免分数归一化问题。"
        }
    ]
