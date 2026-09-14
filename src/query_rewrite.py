"""
查询改写（追问消解）——把多轮追问压成一句自包含的检索 query

为什么需要：
  用户接着上一轮问「那第二点呢」，检索器直接拿这四个字去向量库捞——捞不到东西。
  pages/3 早已把 history 取出来传给 stream_rag_query，但它只被送进生成阶段
  （_prepare_generation 把历史注入 system 与当前问题之间），rag_qa 里检索那一行
  吃的仍是原始字符串。数据本来就在手，只是没接上检索。
  实测症状：拿「那第二点呢」检索 top5，只有 1 条相似度 >0.5，第 3 条起断崖
  （0.6385 → 0.0147，43 倍落差）。

做法（每一步都是「能不做就不做」）：
  1. 无历史（单轮独立提问）→ 直接返回原 query。不调 LLM，零成本零延迟；
  2. 有历史 → 让 LLM 把「历史 + 当前问题」压成一句自包含的检索 query。
     系统提示词约束它只补指代：不扩写、不拆问、不回答、不引入历史外信息；
  3. 任何异常、空结果、超长结果 → 退回原 query。
     与 reranker 同一套降级思路——优化环节允许失效，但不能阻断问答。

代价：
  每次追问多一次 LLM 调用（约 0.3–1s）。追问本来就不走语义缓存，不受影响。
  要关掉做对照：QUERY_REWRITE_ENABLED=false。
"""
import sys

from openai import OpenAI

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, RAG_HISTORY_TURNS,
    QUERY_REWRITE_ENABLED, QUERY_REWRITE_MAX_CHARS,
)

# 单条历史送进提示词前的截断长度：历史只用来消解指代，不需要全文
_HISTORY_CHAR_LIMIT = 300

# 惰性单例：没配 key 的环境（CI / 单测）也能 import 本模块
_client = None


def _get_client() -> OpenAI:
    """取云端 LLM 客户端，首次调用时创建（惰性单例）。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


REWRITE_SYSTEM_PROMPT = """你是知识库检索的查询改写器。

多轮对话里，用户的最新问题常带指代和省略（「它」「那第二点呢」「这个怎么配」），
直接拿去检索会捞不到东西。你的任务是把它改写成一句**不依赖对话历史也能独立看懂**的检索查询。

规则：
1. 只补全指代和省略，不改变原意、不扩大范围。
2. 不要回答问题，不要拆成多个问题，不要补充历史中没有出现过的信息。
3. 最新问题本身已经自包含时，原样输出。
4. 只输出改写后的问题本身：不要解释、不要前缀、不要引号、不要任何额外修饰。"""


def _format_history(history: list[dict]) -> str:
    """把最近若干条历史渲染成「用户：…／助手：…」文本，供提示词使用。"""
    lines = []
    for h in history[-RAG_HISTORY_TURNS:]:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        name = "用户" if role == "user" else "助手"
        lines.append(f"{name}：{content[:_HISTORY_CHAR_LIMIT]}")
    return "\n".join(lines)


def _build_user_prompt(query: str, history: list[dict]) -> str:
    return (
        f"对话历史：\n{_format_history(history)}\n\n"
        f"最新问题：{query}\n\n"
        "改写结果："
    )


# 模型偶尔会画蛇添足加前缀或引号，这里统一剥掉
_PREFIXES = ("改写结果：", "改写结果:", "改写后：", "改写后:", "查询：", "查询:", "检索查询：", "检索查询:")
_QUOTE_CHARS = "\"'“”‘’《》「」「」"


def _clean(text: str) -> str:
    """剥掉可能的前缀/引号，并只保留首个非空行。"""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    t = lines[0]
    for p in _PREFIXES:
        if t.startswith(p):
            t = t[len(p):].strip()
            break
    t = t.strip(_QUOTE_CHARS).strip()
    return t


def _is_usable(rewritten: str) -> bool:
    """结果可用性判定——任何一条不满足都退回原 query。"""
    if len(rewritten) < 2:
        return False
    # 超长基本是模型开始"回答"而不是"改写"（也可能把历史全文抄了回来）
    if len(rewritten) > QUERY_REWRITE_MAX_CHARS:
        return False
    return True


def needs_rewrite(history: list[dict] | None) -> bool:
    """是否需要消解指代：有历史，且历史里有非空的用户消息。

    只有助手消息（异常场景）不算——没有用户提问就无从消解。
    """
    if not history:
        return False
    return any(
        isinstance(h, dict)
        and h.get("role") == "user"
        and (h.get("content") or "").strip()
        for h in history
    )


def rewrite_query(query: str, history: list[dict] | None = None,
                  client=None, model: str | None = None) -> str:
    """把追问改写成自包含的检索 query；不需要 / 失败时原样返回。

    client / model 可注入，便于单测用假客户端跑通全部分支。
    """
    if not QUERY_REWRITE_ENABLED or not needs_rewrite(history):
        return query

    try:
        resp = (client or _get_client()).chat.completions.create(
            model=model or LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, history)},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        rewritten = _clean(resp.choices[0].message.content or "")
    except Exception as e:
        # 降级不静默：写 stderr，但绝不让异常冒泡打断问答
        print(f"[QueryRewrite] 改写失败，退回原问题: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return query

    if not _is_usable(rewritten):
        print(f"[QueryRewrite] 改写结果不可用（len={len(rewritten)}），退回原问题",
              file=sys.stderr, flush=True)
        return query

    return rewritten
