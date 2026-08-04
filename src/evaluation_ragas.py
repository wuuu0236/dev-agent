"""
RAGAS 评估器 —— 用业界标准框架（ragas）跑四维指标，输出 0-100 百分制。

与手写版（src/evaluation.py）的关系：
  · 手写四维（Context Recall / Precision / Faithfulness / Answer Relevancy）
    就是 RAGAS 的核心四指标，这里用 RAGAS 标准化计算（且保留手写版做对照/教学）。
  · 手写版是 1-5 分，RAGAS 输出 0-1，统一 ×100 变百分制展示。

两个 DeepSeek / 硅基流动兼容坑（踩过，别回退）：
  1. DeepSeek API 只接受 n=1。ragas 内部 n>1 的多次生成是通过
     prompts=[prompt]*n 实现的——所以设 bypass_n=True + ChatOpenAI(n=1)，
     API 层永远 n=1，多次生成照常。
  2. OpenAIEmbeddings 默认 check_embedding_ctx_length=True 会把文本先
     tiktoken 成 token 数组再请求，硅基流动 /embeddings 不接受整数数组，
     必须设 False 让 langchain 直接发字符串。

对外 API：
  run_ragas_eval(kb_id, test_questions, top_k=5) -> dict
      {metrics: {name: 0-100 均值, ...}, details: [{question, answer, scores: {...}}]}
  compare_configs(kb_id, test_questions, configs) -> dict
      同一测试集多组配置各跑一遍，用于对比评分（如 top_k 调优）。
"""
import os
import json
import time
import glob
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import _LangchainLLMWrapper
from ragas.embeddings import _LangchainEmbeddingsWrapper
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.config import EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_MODEL, DATA_DIR
from src.rag_agent import rag_query
# 必须在 rag_agent（它会 import langfuse.openai）之后加载，修复 usage=None 崩溃
import src.langfuse_compat  # noqa: E402,F401

load_dotenv()

EVAL_HISTORY_DIR = DATA_DIR / "eval_history"

# 四指标 = 手写版的四维
METRICS = [context_precision, context_recall, faithfulness, answer_relevancy]
METRIC_NAMES = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def _make_llm():
    """ragas 的 LLM wrapper（DeepSeek，兼容 n=1 限制）。"""
    return _LangchainLLMWrapper(
        ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.1,
            n=1,          # DeepSeek 只支持 n=1
        ),
        bypass_n=True,    # 阻止 ragas 把 n 改成 >1
    )


def _make_embeddings():
    """ragas 的 Embedding wrapper（硅基流动，兼容 token 数组限制）。"""
    return _LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY or LLM_API_KEY,
            base_url=EMBEDDING_API_BASE,
            check_embedding_ctx_length=False,  # 硅基流动不接受 token 数组 input
        )
    )


def _to_percent(v) -> float:
    """0-1 → 0-100 百分制（容忍 None/nan）。"""
    try:
        if v is None or (isinstance(v, float) and v != v):  # nan
            return 0.0
        return round(float(v) * 100, 1)
    except (TypeError, ValueError):
        return 0.0


def run_ragas_eval(kb_id: str, test_questions: list[dict], top_k: int = 5) -> dict:
    """
    对一份测试集跑 RAGAS 四指标。

    test_questions: [{"question": "...", "reference": "...", ...}, ...]
    返回：{metrics: {name: 百分制均值}, details: [{question, answer, scores}]}
    """
    if not test_questions:
        return {"error": "测试集为空"}

    samples = []
    details = []

    # 1. 对每题跑 RAG 管线（检索 + 生成），收集 answer 和 retrieved_contexts
    for item in test_questions:
        result = rag_query(kb_id, item["question"], top_k=top_k)
        contexts = [c["content"] for c in result.get("contexts", [])]
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=item["reference"],
        ))
        details.append({
            "question": item["question"],
            "answer": result["answer"],
        })

    # 2. RAGAS 打分（LLM judge + 语义相似度）
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=list(METRICS),
        llm=_make_llm(),
        embeddings=_make_embeddings(),
        show_progress=False,
        raise_exceptions=False,  # 单题失败不拖垮整批，分数记 0
    )

    # 3. 逐题分数 + 均值
    rows = result.to_pandas()
    metrics = {}
    for name in METRIC_NAMES:
        col = rows[name].tolist()
        metrics[name] = _to_percent(sum(col) / len(col) if col else 0)

    for i, detail in enumerate(details):
        detail["scores"] = {name: _to_percent(rows[name].iloc[i]) for name in METRIC_NAMES}

    return {"metrics": metrics, "details": details}


# ================================================================
# 对比 harness：同一测试集、多组配置各跑一遍（对比评分）
# ================================================================

def compare_configs(kb_id: str, test_questions: list[dict],
                    configs: list[dict], save: bool = True) -> dict:
    """
    同一测试集、多组检索配置各跑一遍 RAGAS，输出对比结果。

    configs: [{"name": "top_k=3", "top_k": 3}, {"name": "top_k=5", "top_k": 5}, ...]
    返回：{timestamp, kb_id, test_size, results: [{name, config, metrics, details}]}
    """
    if not configs:
        return {"error": "配置为空"}

    results = []
    for cfg in configs:
        top_k = cfg.get("top_k", 5)
        res = run_ragas_eval(kb_id, test_questions, top_k=top_k)
        results.append({
            "name": cfg["name"],
            "config": cfg,
            "metrics": res.get("metrics", {}),
            "details": res.get("details", []),
        })

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kb_id": kb_id,
        "test_size": len(test_questions),
        "results": results,
    }

    if save:
        save_result(out)
    return out


# ================================================================
# 结果持久化：每次对比存档，面板可回看历史
# ================================================================

def save_result(result: dict) -> str:
    """把一次对比结果存成 JSON，返回文件名。"""
    EVAL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = EVAL_HISTORY_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return fname


def list_history() -> list[dict]:
    """列出所有历史对比存档（倒序，最新在前）。"""
    files = sorted(glob.glob(str(EVAL_HISTORY_DIR / "eval_*.json")), reverse=True)
    out = []
    for f in files:
        try:
            data = json.loads(open(f, encoding="utf-8").read())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "file": os.path.basename(f),
            "timestamp": data.get("timestamp", ""),
            "kb_id": data.get("kb_id", ""),
            "test_size": data.get("test_size", 0),
            "configs": [r["name"] for r in data.get("results", [])],
        })
    return out


def load_history(filename: str) -> dict | None:
    """加载一份历史对比结果。"""
    path = EVAL_HISTORY_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
