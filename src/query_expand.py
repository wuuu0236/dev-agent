"""
查询扩展（Multi-Query Expansion）：把一个检索 query 变成多个不同角度的
等价表述，各查一路再融合，弥补"单一表述召不全"的问题。

与 query_rewrite 的关系（两步各管一件事，先后有序）：
  · query_rewrite 解决「指代不完整」——"那第二点呢" → 自包含的完整问题，先执行；
  · query_expand 解决「表述单一」——同一个信息需求换几个角度各召回一次，后执行。

成本与代价（为什么默认关）：
  · 每问多一次 LLM 调用（生成扩展表述）；
  · 每路扩展各跑一遍粗排（向量 + BM25），但**精排仍只做一次**（见
    HybridRetriever.search_multi）——账单增量主要是 embedding 次数，可忽略。
  开关 `QUERY_EXPANSION_ENABLED`，默认 false。

失败语义：扩展失败 / 关闭时返回 [原始 query]，问答链路照常走单查询路径——
扩展是锦上添花，绝不能因为它把问答搞挂。
"""
import sys
from openai import OpenAI

from src.config import LLM_MODEL, QUERY_EXPANSION_COUNT, QUERY_EXPANSION_ENABLED

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from src.config import LLM_API_KEY, LLM_BASE_URL
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


EXPAND_SYSTEM_PROMPT = (
    "你是一个检索查询扩展器。我会给你一个用户问题，你需要把它改写成 {count} 个不同角度的等价检索查询。\n"
    "规则：每个查询都是独立完整的检索语句。每个查询从不同角度表述同一个信息需求。\n"
    "不要回答问题，不要解释，不要输出编号。每行一个查询，只输出查询文本。\n"
    "如果问题很短或已经很通用，输出原问题 {count} 次。"
)


def expand_query(query: str) -> list[str]:
    """把一个 query 扩展为多个等价查询。关闭 / 失败时返回 [query]。

    原始 query 永远排在第一位（主查询：去重保序 + 精排打分都以它为准），
    LLM 的扩展表述跟在后面；与原始重复的表述会被去掉。
    """
    if not QUERY_EXPANSION_ENABLED:
        return [query]
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXPAND_SYSTEM_PROMPT.format(count=QUERY_EXPANSION_COUNT)},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        lines = [line.strip() for line in
                 response.choices[0].message.content.strip().split("\n") if line.strip()]
        # dict.fromkeys 保序去重：原始 query 打头，LLM 表述去掉与原始重复的行
        return list(dict.fromkeys([query] + lines))[:QUERY_EXPANSION_COUNT]
    except Exception as e:
        print(f"[QueryExpand] 扩展失败退回原始 query: {e}", file=sys.stderr)
        return [query]
