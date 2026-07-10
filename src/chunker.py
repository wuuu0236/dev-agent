"""
文本切块器

为什么需要 chunk：
  - LLM 上下文有限，不能一次塞进整本书
  - 太长的文本包含太多无关信息，干扰 LLM 判断
  - 检索粒度越细，越容易找到「那一小段」精确答案

为什么 chunk_size=500：
  - 中文 3-5 句话大约是 300-600 字，500 是一个完整的「信息单元」
  - 太小（200）：一句话被切碎，丢失上下文
  - 太大（2000）：一个块塞进太多无关内容，检索不准

为什么 overlap=50：
  - 防止关键信息刚好落在两个 chunk 的分界线上
  - 50 字大约覆盖一个句子的末尾 + 下一个句子的开头
  - 保证关键信息至少在一个完整 chunk 里出现
"""
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    滑动窗口切片。

    例如 chunk_size=500, overlap=50:
      块1: text[0:500]
      块2: text[450:950]   ← 和块1共享 text[450:500]
      块3: text[900:1400]
      ...

    对中文额外处理：优先在句号、换行处切分，尽量不切断句子。
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # 如果没到文本末尾，尝试在句子边界切分
        if end < len(text):
            # 从 end 位置往回找最近的句子分隔符
            search_start = max(start, end - 100)  # 最多往回找 100 字
            best_end = end
            for sep in ["\n\n", "\n", "。", "；", "，", ".", ";", ","]:
                # 在 [search_start, end] 范围内找分隔符
                pos = text.rfind(sep, search_start, end)
                if pos != -1:
                    best_end = pos + len(sep)  # 切在分隔符之后
                    break
            end = best_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 下一个块的起点 = 当前终点 - overlap
        start = end - overlap
        if start >= len(text):
            break

    return chunks


def chunk_parsed(parsed_docs: list[dict]) -> list[dict]:
    """
    把解析后的文档列表切成 chunk 列表。

    输入：[{text, page, source}, ...]  ← parse_file 的输出
    输出：[{content, page, source, chunk_index}, ...]
    """
    chunks = []
    for doc in parsed_docs:
        text_parts = split_text(doc["text"])
        for i, part in enumerate(text_parts):
            chunks.append({
                "content": part,
                "page": doc.get("page"),
                "source": doc["source"],
                "chunk_index": i
            })
    return chunks
