"""
Contextual Retrieval：给每个 chunk 生成一句「这个片段在整篇文档里的上下文」，
拼在 chunk 内容前面再嵌入。

为什么需要：
  滑窗切出来的 chunk 经常是"半截语境"——「循环一直持续」放在哪个问题下看都像相关。
  标题树模式能拼标题路径，但那只是文字前缀、不是语义描述，且 PDF / OCR 场景
  标题路径经常为空。Anthropic 的实验结论：给每个 chunk 配一句文档级上下文，
  检索失败率显著下降。

成本与代价（为什么默认关）：
  · 入库时**每个 chunk 一次 LLM 调用**——83 个 chunk 的文档就是 83 次调用，
    入库时间与账单都随 chunk 数线性增长；
  · 检索阶段零额外开销（前缀在入库时已拼好，跟着向量一起存）。
  开关 `CONTEXTUAL_RETRIEVAL_ENABLED`，默认 false。

设计要点：
  · 客户端惰性构造（`_get_client`）——模块导入不得要求凭据（CI 约定，见
    tests/test_import_no_credentials.py）；
  · 单块失败只影响那一个块（不加前缀），绝不打断整批入库；
  · 模型自认「自成一体」时返回空串——图片 OCR 块、自包含段落不该被硬加前缀。
"""
import sys
from openai import OpenAI

from src.config import (
    CONTEXTUAL_DOCUMENT_MAX_CHARS,
    CONTEXTUAL_MODEL,
    CONTEXTUAL_RETRIEVAL_ENABLED,
)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from src.config import LLM_API_KEY, LLM_BASE_URL
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


SYSTEM_PROMPT = (
    "你是一个文档上下文生成器。我会给你一份文档的内容片段和这篇文档的完整文本（可能被截断）。\n"
    "你的任务：用 1-2 句话描述这个片段在整篇文档中的位置和语境。\n"
    "规则：只描述片段在文档中的位置和主题，不要回答片段中的问题。\n"
    "不要引入文档中没有的信息。只输出描述文字。\n"
    "如果片段本身已经自包含输出：此片段自成一体，无需额外上下文。"
)


def generate_context(chunk_text: str, full_document: str) -> str:
    """给一个 chunk 生成上下文描述。关闭 / 失败 / 片段自包含时返回空串。"""
    if not CONTEXTUAL_RETRIEVAL_ENABLED:
        return ""
    truncated_doc = full_document[:CONTEXTUAL_DOCUMENT_MAX_CHARS]
    user_prompt = (
        f"<document>\n{truncated_doc}\n</document>\n\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n\n上下文描述："
    )
    try:
        response = _get_client().chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        result = response.choices[0].message.content.strip()
        if "自成一体" in result:
            return ""
        return result
    except Exception as e:
        print(f"[Contextualizer] 生成上下文失败: {e}", file=sys.stderr)
        return ""


def contextualize_chunks(chunks: list[dict], full_document: str) -> list[dict]:
    """批量处理：给每个 chunk 的 content 前面拼上 `[上下文] ` 前缀。

    开关关闭时整体 no-op（原样返回）。某个块生成失败只影响它自己——
    不加前缀照常入库，绝不让一个 LLM 调用失败拖垮整批文档。
    """
    if not CONTEXTUAL_RETRIEVAL_ENABLED:
        return chunks
    for chunk in chunks:
        context = generate_context(chunk["content"], full_document)
        if context:
            chunk["content"] = f"[{context}] {chunk['content']}"
    return chunks


def build_full_document(parsed_docs: list[dict]) -> str:
    """把 parse_file 的输出拼成「整篇文档」文本，供上下文生成用。

    为什么不按文档说的给 parse_file 加 full_text 字段：parse_file 的返回结构
    已被多处消费（含既有测试断言），为一个模块的需求改公共契约不划算——
    在调用侧把各段的 text 拼起来，语义完全等价。
    """
    return "\n\n".join(d.get("text", "") for d in parsed_docs if d.get("text"))
