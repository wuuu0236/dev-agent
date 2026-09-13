"""
Rerank 精排（交叉编码 / cross-encoder）

为什么需要它 —— 双塔 vs 交叉编码：

  · **向量检索 = 双塔（bi-encoder）**
    query 和 doc **各自独立**编码成向量，再算余弦相似度。
    query 编码时完全不知道要跟哪篇 doc 比，doc 编码时也不知道会被谁查。
    好处：doc 向量可以离线算好、建索引，查询时只算 query 一次 → 快，能扛百万级。
    代价：**两者从未见过面**，丢失了词级交互信息。"这个问题"和"这个答案"
    字面上可能毫无重叠，双塔只能靠整体语义的粗糙对齐来判断。

  · **Rerank = 交叉编码（cross-encoder）**
    把 query 和 doc **拼成一条**送进模型，全程做注意力交互，直接输出一个
    相关性分数。准得多，但每条候选都要单独跑一遍模型 → 慢，且无法预建索引。
    代价决定了它只能用在**小候选集**上。

所以工程架构是「粗排 → 精排」：
    query ──双塔──> 候选池 top_k×10（快、粗、保证召回）
                          │
                          └──交叉编码──> 最终 top_k（慢、准、保证排序）

  这个"用便宜的方法缩小范围、用贵的方法做最终判断"的分层，是几乎所有
  检索系统的通用结构（搜索引擎、推荐系统同理）。

一个额外的好处 —— 分数有绝对意义：
  实测 bge-reranker-v2-m3：相关文档 0.86，不相关文档 0.00002。
  而 RRF 融合分是 1/(60+rank)，量级完全由排名决定、与"有多相关"无关。
  所以想做「相似度低于阈值就拒答」，**阈值必须建在 rerank 分数上**，
  建在 RRF 分数上是没有意义的。

行为约定：
  · 失败即降级——API 超时/报错/未配置时，原样返回粗排结果的前 top_k，
    绝不让精排挂掉整个问答链路。
  · 可用 RERANK_ENABLED=false 关闭做 A/B 对比。
"""
import sys

import httpx

from src.config import (
    RERANK_API_BASE,
    RERANK_API_KEY,
    RERANK_ENABLED,
    RERANK_MAX_DOC_CHARS,
    RERANK_MODEL,
    RERANK_TIMEOUT,
)

# 单次请求送审的文档条数上限。分批的理由不只是 API 限制——
# 一次性送 50 条长文本会让单请求变大、更容易超时；分批后失败影响面也小。
_RERANK_BATCH = 32


def _log(msg: str) -> None:
    print(f"[Rerank] {msg}", file=sys.stderr, flush=True)


def _score_batch(query: str, docs: list[str]) -> list[float] | None:
    """调一次 rerank API，返回与 docs 等长的分数列表；失败返回 None。"""
    url = RERANK_API_BASE.rstrip("/") + "/rerank"
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {RERANK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": docs,
                "top_n": len(docs),
                "return_documents": False,
            },
            timeout=RERANK_TIMEOUT,
        )
        if resp.status_code != 200:
            _log(f"API 返回 {resp.status_code}: {resp.text[:200]}")
            return None
        results = resp.json().get("results") or []
        if len(results) != len(docs):
            _log(f"返回条数不符：期望 {len(docs)}，实得 {len(results)}")
            return None
        # API 返回按分数降序的 [{index, relevance_score}]，还原成与输入同序
        scores = [0.0] * len(docs)
        for item in results:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs):
                scores[idx] = float(item.get("relevance_score") or 0.0)
        return scores
    except Exception as e:
        _log(f"调用失败，降级为粗排顺序: {type(e).__name__}: {e}")
        return None


def rerank(query: str, candidates: list[dict], top_k: int,
           enabled: bool | None = None) -> list[dict]:
    """对粗排候选集做交叉编码精排，返回前 top_k 条。

    candidates: [{content, source, page, ...}, ...]（HybridRetriever 的 RRF 输出）
    enabled:    显式覆盖开关（评测里做 A/B 用），None 则读 config

    返回的每条会多一个 rerank_score 字段（0~1，越大越相关）。
    任何异常都降级为「原顺序前 top_k」，并保留原有字段。
    """
    if not candidates:
        return []

    use = RERANK_ENABLED if enabled is None else enabled
    if not use:
        return candidates[:top_k]

    if not RERANK_API_KEY:
        _log("未配置 RERANK_API_KEY，跳过精排")
        return candidates[:top_k]

    # 截断保护：chunk 通常 500 字，但多模态/异常块可能超长
    docs = [(c.get("content") or "")[:RERANK_MAX_DOC_CHARS] for c in candidates]

    scores: list[float] = []
    for start in range(0, len(docs), _RERANK_BATCH):
        part = _score_batch(query, docs[start:start + _RERANK_BATCH])
        if part is None:
            return candidates[:top_k]   # 任一批失败即整体降级，保证顺序一致性
        scores.extend(part)

    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    ranked: list[dict] = []
    for i in order[:top_k]:
        item = dict(candidates[i])
        item["rerank_score"] = round(scores[i], 6)
        ranked.append(item)
    return ranked
