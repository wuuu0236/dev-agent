"""
文本切块器

链路位置：解析 → 清洗 → **分块** → 嵌入 → 入库

两种策略：
  1. **标题树分块（默认，Markdown/Word 转 md 场景）**——对标 MaxKB
     按 `#`~`######` 把文档切成章节，每个 chunk 前面拼上"父路径"：

         Agent 基础原理 > ReAct 模式
         循环一直持续，直到模型输出最终答案...

     为什么这么干：chunk 脱离文档后是"孤儿"，"循环一直持续"这种句子单独看
     不知道在说什么。拼上标题路径后，这一段自带上下文，向量语义更完整，
     召回率和可读性同时变好（人看引用时也知道这坨内容出自哪一节）。

     章节合并：相邻短章节打包到一块（<= CHUNK_SIZE），包内多章时把各自的
     标题写回正文，信息不丢；只有一章且超长 → 交给滑动窗口再切，每片带前缀。

  2. **滑动窗口（无标题时的兜底：PDF 页文本、OCR、纯 TXT）**
     见 split_text。

为什么 chunk_size=500：
  - 中文 3-5 句话大约是 300-600 字，500 是一个完整的「信息单元」
  - 太小（200）：一句话被切碎，丢失上下文
  - 太大（2000）：一个块塞进太多无关内容，检索不准

为什么 overlap=50：
  - 防止关键信息刚好落在两个 chunk 的分界线上
  - 50 字大约覆盖一个句子的末尾 + 下一个句子的开头

可用 CHUNK_TREE_ENABLED=false 退回纯滑窗做 A/B 对比。

碎块合并（两种策略共用，收口在 chunk_parsed）：
  滑窗的尾块、标题树末尾的短章节，都可能只剩几十个字。这类碎块语义不完整，
  检索时容易误命中，还白占一个候选位。长度 < CHUNK_MIN_SIZE（默认 120）的块
  会并入前一块（首块则并入后一块）。设 0 可关闭。
"""
import re

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MIN_SIZE,
    CHUNK_PATH_SEP,
    CHUNK_PATH_MAXLEN,
    CHUNK_TREE_ENABLED,
)

# Markdown 标题：行首 1-6 个 # + 空格
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 代码围栏（围栏内的 # 是注释，不是标题，必须屏蔽）
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


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


# ---------------------------------------------------------------- 标题树分块


def _parse_sections(text: str) -> list[dict]:
    """按 Markdown 标题把文本切成章节。

    返回 [{"path": ["一级", "二级"], "body": "正文"}, ...]
    path 是该章节的完整标题路径；正文不含标题行本身。
    代码围栏内的 `#` 不当标题（否则 Python 注释会被误判成章节）。
    """
    sections: list[dict] = []
    stack: list[str] = []      # 当前标题栈，stack[i] 是第 i+1 级标题
    body: list[str] = []
    in_code = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append({"path": list(stack), "body": content})
        body.clear()

    for raw in text.split("\n"):
        if _FENCE_RE.match(raw):
            in_code = not in_code
            body.append(raw)
            continue
        if not in_code:
            m = _HEADING_RE.match(raw)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip().strip("#").strip()
                while len(stack) >= level:   # 回退到上一级
                    stack.pop()
                stack.append(title)
                continue
        body.append(raw)

    flush()
    return sections


def _common_prefix(paths: list[list[str]]) -> list[str]:
    """多个标题路径的最长公共前缀。"""
    if not paths:
        return []
    prefix = list(paths[0])
    for p in paths[1:]:
        i = 0
        while i < len(prefix) and i < len(p) and prefix[i] == p[i]:
            i += 1
        prefix = prefix[:i]
    return prefix


def _format_path(path: list[str]) -> str:
    """拼成 'A > B > C'；过长时丢掉最前面的层级（保留最近的上下文）。"""
    if not path:
        return ""
    text = CHUNK_PATH_SEP.join(path)
    if len(text) <= CHUNK_PATH_MAXLEN:
        return text
    kept: list[str] = []
    for title in reversed(path):                      # 从后往前尽量塞
        candidate = [title, *kept]
        if len(CHUNK_PATH_SEP.join(candidate)) > CHUNK_PATH_MAXLEN:
            break
        kept = candidate
    if not kept:                                      # 单条标题就超长
        kept = [path[-1][:CHUNK_PATH_MAXLEN]]
    return ("…" + CHUNK_PATH_SEP if len(kept) < len(path) else "") + CHUNK_PATH_SEP.join(kept)


def _group_sections(sections: list[dict], chunk_size: int) -> list[list[dict]]:
    """相邻章节打包：累计不超过 chunk_size 的并进同一块；超长章节独占一块。"""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0

    for sec in sections:
        size = len(sec["body"])
        if size > chunk_size:
            if cur:
                groups.append(cur)
                cur, cur_len = [], 0
            groups.append([sec])          # 单章超长，后面交给滑动窗口
            continue
        if cur and cur_len + size > chunk_size:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(sec)
        cur_len += size

    if cur:
        groups.append(cur)
    return groups


def _render_group(group: list[dict]) -> str:
    """把一组章节渲染成一段正文：公共路径前缀 + 各章（补回被折叠的标题）。"""
    prefix = _common_prefix([s["path"] for s in group])
    depth = len(prefix)

    parts = []
    for sec in group:
        extra = sec["path"][depth:]
        head = "".join(f"{'#' * (depth + i + 1)} {t}\n" for i, t in enumerate(extra))
        parts.append(head + sec["body"])
    body = "\n\n".join(parts).strip()

    path_text = _format_path(prefix)
    return f"{path_text}\n{body}" if path_text else body


def chunk_text_tree(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """标题树分块：带父路径前缀；无标题时自动退回滑动窗口。"""
    if not text or not text.strip():
        return []

    sections = _parse_sections(text)
    if not sections:
        return []

    # 整篇没有一个标题（PDF 页文本 / OCR / 纯 TXT）→ 退回滑窗，行为与旧版一致
    if not any(s["path"] for s in sections):
        return split_text(text, chunk_size, overlap)

    chunks: list[str] = []
    for group in _group_sections(sections, chunk_size):
        rendered = _render_group(group)
        if len(rendered) <= chunk_size:
            chunks.append(rendered)
        else:
            # 单章超长：滑动窗口再切，每一片都保留该章的路径前缀
            path_line = rendered.split("\n", 1)[0]
            rest = rendered.split("\n", 1)[1] if "\n" in rendered else ""
            for piece in split_text(rest or rendered, chunk_size, overlap):
                chunks.append(f"{path_line}\n{piece}")
    return [c for c in chunks if c.strip()]


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """统一入口：按开关决定用标题树还是滑动窗口。"""
    if CHUNK_TREE_ENABLED:
        return chunk_text_tree(text, chunk_size, overlap)
    return split_text(text, chunk_size, overlap)


def merge_tiny_chunks(chunks: list[str], min_size: int = CHUNK_MIN_SIZE) -> list[str]:
    """把过短的 chunk 并入相邻块，消掉"只有一两句话"的孤立碎块。

    碎块从哪来：
      · 滑窗的尾块——切到文末时剩下多少就是多少，可能只剩几十个字
      · 标题树里最后一个短章节——打包逻辑只负责"累计到 500"，末尾不满也照样成块

    为什么必须处理：
      1. 碎块语义不完整，检索时容易误命中——"循环一直持续"这种半截句子
         放在哪个问题下看都像相关，会挤掉真正完整的那一块
      2. 白白占用一个候选位，等于提高了召回难度

    合并方向选"并入前一块"而非后一块：前一块通常是同一章节的延续，
    语义上更近；且拼完仍保持自然阅读顺序。

    首块本身过短时没有"前一块"可并，改为往后并。
    """
    if min_size <= 0 or len(chunks) <= 1:
        return chunks

    out: list[str] = []
    for c in chunks:
        if out and len(c) < min_size:
            out[-1] = f"{out[-1]}\n{c}"
        else:
            out.append(c)

    # 首块过短 → 只能往后并
    if len(out) >= 2 and len(out[0]) < min_size:
        out[1] = f"{out[0]}\n{out[1]}"
        out.pop(0)

    return out


# ---------------------------------------------------------------- 对外主入口


def chunk_parsed(parsed_docs: list[dict]) -> list[dict]:
    """
    把解析后的文档列表切成 chunk 列表。

    输入：[{text, page, source, type?, image?}, ...]  ← parse_file 的输出（已清洗）
    输出：[{content, page, source, chunk_index, type, image}, ...]
          type/image 透传，供多模态检索与视觉模型读取原图。
    """
    chunks = []
    for doc in parsed_docs:
        text_parts = chunk_document(doc["text"])
        # 收口在这里做碎块合并：两种策略（标题树/滑窗）都经过本函数，一处生效
        text_parts = merge_tiny_chunks(text_parts, CHUNK_MIN_SIZE)
        doc_type = doc.get("type", "text")
        doc_image = doc.get("image")
        for i, part in enumerate(text_parts):
            chunks.append({
                "content": part,
                "page": doc.get("page"),
                "source": doc["source"],
                "chunk_index": i,
                "type": doc_type,
                "image": doc_image,
            })
    return chunks
