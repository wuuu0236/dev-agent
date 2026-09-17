"""
CRAG 生成后自检：LLM 生成答案后，让模型判断「这个答案是否真的基于给出的上下文」。

与 answer_gate（检索侧门控）的分工：
  · answer_gate 管**进门**——检索分数太低就不把上下文喂给模型（防幻觉上下文）；
  · answer_verifier 管**出门**——答案已经生成了，回头检查它是否真的基于上下文，
    不达标就触发更严格的重新生成（CRAG 的 corrective 分支）。

失败语义：**失败即放行**（返回 5 分）。验证的目的是提高答案质量，
不能反过来变成阻断回答的新故障点——验证器挂了，问答链路必须照常工作。

成本：开启后每问多一次 LLM 调用（max_tokens=5 的打分调用，很便宜）；
只有分数低于阈值时才额外付一次「严格重试」的完整生成调用（失败路径才触发）。
"""
import sys
from openai import OpenAI

from src.config import ANSWER_VERIFY_ENABLED, LLM_MODEL

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from src.config import LLM_API_KEY, LLM_BASE_URL
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


VERIFY_SYSTEM_PROMPT = (
    "你是一个 RAG 答案验证器。我会给你一段上下文和一个基于该上下文生成的答案。\n"
    "判断答案是否完全基于上下文中的信息。评分标准 1-5：\n"
    "5 = 每个陈述都能在上下文中找到直接依据\n"
    "4 = 基本基于上下文有少量合理推断\n"
    "3 = 部分基于上下文部分内容来源不明\n"
    "2 = 有明显的编造或与上下文矛盾\n"
    "1 = 完全脱离上下文\n"
    "只输出一个数字 1-5 不要解释。"
)


def verify_answer(answer: str, contexts: list[str]) -> int:
    """返回 grounded 分数 1-5。关闭 / 无上下文 / 验证失败时返回 5（默认放行）。"""
    if not ANSWER_VERIFY_ENABLED or not contexts:
        return 5
    context_text = "\n---\n".join(contexts[:5])[:4000]
    user_prompt = f"<context>\n{context_text}\n</context>\n\n<answer>\n{answer}\n</answer>\n\n分数："
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip().rstrip(".")
        score = int(raw)
        return max(1, min(5, score))
    except Exception as e:
        print(f"[AnswerVerifier] 验证失败默认放行: {e}", file=sys.stderr)
        return 5
