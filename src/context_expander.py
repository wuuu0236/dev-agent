"""
Parent-Child Retrieval（docs 第三章）：检索用小块、生成用大块。

小块（500 字）嵌入检索准，但喂给 LLM 太碎——答案容易缺前后文。
精排命中后按 source + chunk_index 把相邻块拉回来合并成邻域大块再给模型，
检索的精准和生成的完整两头都占到。不需要任何数据库变更：
metadata 里本来就存着 chunk_index（本次修了透传：search_similar /
get_all_chunks / BM25 路此前都没把它带出来）。

引用展示仍用**精排命中的原始 chunk**：合并文本是给模型的，不是给用户看的——
用户展开引用要核实的是"命中了哪句话"，不是整段邻域。因此扩展时把原始
content 存进 original_content，citations 模块优先取它。

失败/边界语义：范围没拉到邻居、老索引缺 chunk_index → 该块原样保留；
开关关闭 → 整体原样返回。扩展是提质手段，绝不让它改变可答性。
"""
from src.config import (
    CONTEXT_EXPAND_MAX_CHARS,
    CONTEXT_EXPAND_NEIGHBORS,
    CONTEXT_EXPANSION_ENABLED,
)


def expand_contexts(hits: list[dict], kb_id: str) -> list[dict]:
    """扩展每个命中 chunk 的上下文邻域。相邻命中合并去重，失败处原样保留。

    返回新列表：扩展的条目 content 变为邻域合并文本、original_content 存原始
    chunk；未扩展的条目原样（引用侧拿到什么都能正确展示）。
    """
    if not CONTEXT_EXPANSION_ENABLED or not hits:
        return hits
    from src.vector_store import get_chunks_by_source_range  # 惰性：不扩展不拖 chromadb

    expanded = []
    seen_ranges: set[tuple] = set()
    covered: dict[str, set] = {}  # source -> 已被前面命中的邻域覆盖的 chunk_index
    for hit in hits:
        source = hit.get("source")
        idx = hit.get("chunk_index")
        # 老索引 / 异常块没有 chunk_index：不扩展，原样保留（保底不丢数据）
        if not source or idx is None:
            expanded.append(hit)
            continue

        if idx in covered.get(source, ()):
            # 这条命中已被前面命中的邻域覆盖：不再重复扩展，但**保留这条命中**
            # （喂给 LLM 两份高度重叠的邻域是纯冗余；丢掉这条则引用来源会
            # 凭空少一条带分数的精排结果——所以原样保留，引用侧按来源去重）
            expanded.append(hit)
            continue

        start = max(0, idx - CONTEXT_EXPAND_NEIGHBORS)
        end = idx + CONTEXT_EXPAND_NEIGHBORS
        key = (source, start, end)
        if key in seen_ranges:
            expanded.append(hit)
            continue
        seen_ranges.add(key)

        neighbors = get_chunks_by_source_range(kb_id, source, start, end)
        if not neighbors:
            expanded.append(hit)
            continue

        covered.setdefault(source, set()).update(range(start, end + 1))
        merged_text = "\n".join(n["content"] for n in neighbors)
        if len(merged_text) > CONTEXT_EXPAND_MAX_CHARS:
            merged_text = merged_text[:CONTEXT_EXPAND_MAX_CHARS] + "..."
        expanded.append({**hit, "content": merged_text,
                         "original_content": hit.get("content", "")})
    return expanded
