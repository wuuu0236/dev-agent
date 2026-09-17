"""
查询路由（Adaptive Routing）：检索之前判断问题类型，决定走哪条路径。

三层结构：规则引擎（零成本，拦最明显的闲聊）→ LLM 分类（兜底判断）→
多跳分解（只对 multi-hop 触发）。

  · chitchat  —— 闲聊/寒暄：不走 RAG（省下 embedding + BM25 + 精排整条链路），
                 由 NO_CONTEXT 提示词正常回应（拒答 ≠ 拒绝服务）
  · simple    —— 单跳：正常检索
  · multi-hop —— 需要组合多个事实才能回答：拆成子问题后联合检索
                 （复用 HybridRetriever.search_multi，精排仍只做一次）

与 answer_gate 的分工：路由管「要不要检索、怎么检索」（检索之前），
门控管「检索结果配不配回答」（检索之后）——路由放行的 simple 问题，
检索出来分数不够照样被门控拦下，两层互不替代。

失败语义：分类失败按 simple 处理、分解失败退回原问题——路由是优化手段，
它挂了问答链路必须照常工作（与 query_expand / answer_verifier 同一原则）。
"""
import re
import sys
from openai import OpenAI

from src.config import ADAPTIVE_ROUTING_ENABLED, LLM_MODEL

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from src.config import LLM_API_KEY, LLM_BASE_URL
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


# 规则引擎：只拦"一眼就是闲聊"的输入。
# ⚠️ 天气/时间/日期类关键词必须锚定句首——「上传时间是什么时候」这类正经
# 知识库问题同样含"时间"二字，不锚定会把它们误杀成 chitchat，
# 误杀的代价是正经问题拿不到检索结果（比漏放一条闲聊严重得多）。
_CHITCHAT_PATTERNS = [
    re.compile(r"^(你好|您好|hi|hello|嗨|哈喽)", re.I),
    re.compile(r"^(谢谢|多谢|thanks|thank you)", re.I),
    re.compile(r"^(今天|昨天|明天|现在)(的)?(天气|时间|日期|几点)"),
    re.compile(r"^(你是谁|你叫什么|你能做什么|你会什么)"),
    re.compile(r"^(帮我写|帮我画|帮我做)(?!(.*(?:文档|知识库|资料)))", re.I),
]

CLASSIFY_SYSTEM_PROMPT = (
    "判断这个问题的类型，只输出一个词：\n"
    "chitchat = 闲聊、寒暄、常识，与知识库内容无关\n"
    "simple = 单一事实查询，一次检索就能回答\n"
    "multi-hop = 需要组合两个或以上事实才能回答\n"
    "只输出 chitchat、simple、multi-hop 之一。"
)

DECOMPOSE_SYSTEM_PROMPT = (
    "把一个多跳问题拆成按顺序回答的子问题列表。\n"
    "每个子问题独立可检索，后一个可以依赖前一个的答案。\n"
    "每行一个子问题，不要编号，不要解释。如果问题不是多跳的，原样输出。"
)


def classify_query(query: str) -> str:
    """判断问题类型，返回 chitchat / simple / multi-hop。

    开关关闭时直接返回 simple（零成本，行为与路由不存在时一致）；
    规则引擎命中 chitchat 不花 LLM 调用；其余交给 LLM 分类兜底。
    """
    if not ADAPTIVE_ROUTING_ENABLED:
        return "simple"
    for pattern in _CHITCHAT_PATTERNS:
        if pattern.search(query):
            return "chitchat"
    return _llm_classify(query)


def _llm_classify(query: str) -> str:
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().lower()
        return result if result in ("chitchat", "simple", "multi-hop") else "simple"
    except Exception as e:
        print(f"[QueryRouter] 分类失败按 simple 处理: {e}", file=sys.stderr)
        return "simple"


def decompose_query(query: str) -> list[str]:
    """把多跳问题拆成子问题列表。失败 / 不是多跳时返回 [query]。"""
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        lines = [line.strip() for line in
                 response.choices[0].message.content.strip().split("\n") if line.strip()]
        return lines if lines else [query]
    except Exception as e:
        print(f"[QueryRouter] 多跳分解失败退回原问题: {e}", file=sys.stderr)
        return [query]
