"""
cleaner.py —— 文档清洗（插在 parser 与 chunker 之间）

为什么需要它：
  解析出来的原文带着一堆"排版噪声"，这些噪声会一路带进 chunk 和向量里：
    · PDF 每行末尾的硬换行 → 句子被切断，语义不完整
    · 每页重复的页眉页脚（"第 X 页""XX公司"）→ 一堆 chunk 含相同噪声，污染检索
    · 连续空行、控制字符、全角空格 → 白白占用 token，干扰分词
    · 页码、单字符等垃圾块 → 单独成 chunk，稀释有效内容
  清洗不产生新知识，但能让 embedding 拿到干净完整的句子——**这是检索质量的地基**。

设计原则：
  1. 只删"确定是噪声"的东西，不做激进改写（宁可少洗，不可洗错）
  2. 中文优先：断行合并按中文标点判断，不破坏英文单词
  3. 可关闭：CLEANER_ENABLED=false 时全部函数原样返回，方便 A/B 对比

用法：
  from src.cleaner import clean_parsed
  parsed = clean_parsed(parse_file(path))   # 跨页信息用于识别页眉页脚
"""
import os
import re
from typing import Iterable

CLEANER_ENABLED = os.getenv("CLEANER_ENABLED", "true").lower() == "true"

# 中文句子终止符（行尾是这些就不与下一行合并）
_SENTENCE_END = "。！？；…”）】》」』!?;:"
# 行首出现这些标记时，不与上一行合并（列表、标题、表格、引用、代码块）
_NO_MERGE_PREFIX = ("#", "-", "*", "+", ">", "|", "```", "~~~", "\t")
# 列表项（1. / 一、/ （1） 等）
_LIST_RE = re.compile(r"^\s*(\d+[.、)]|[一二三四五六七八九十]+[、.)]|[(（]\d+[)）])")
# 控制字符（保留 \n \t \r）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 连续 3 个以上换行
_BLANK_RE = re.compile(r"\n{3,}")
# 行内连续空白（不含换行）
_SPACE_RE = re.compile(r"[ \t\u3000]{2,}")

# 页眉页脚判定：出现页数占比 >= 该值，且长度 <= 该字数
_HEADER_FOOTER_MIN_RATIO = float(os.getenv("CLEANER_HF_RATIO", "0.5"))
_HEADER_FOOTER_MAX_LEN = int(os.getenv("CLEANER_HF_MAXLEN", "30"))


def _is_numbered_or_pure_number(line: str) -> bool:
    """纯数字行（页码）或极短无意义行。"""
    s = line.strip()
    if not s:
        return True
    return bool(re.fullmatch(r"[-–—.\s\d]{1,10}", s))


def normalize(text: str) -> str:
    """基础归一化：控制字符、换行、行尾空白、连续空行。"""
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def merge_broken_lines(text: str) -> str:
    """合并被硬换行切断的句子（治 PDF 每行一个 \\n 的老毛病）。

    规则：上一行**不是**以句末标点结尾、且下一行**不是**标题/列表/表格/代码块开头时，
    两行合并。英文单词之间补一个空格，中文直接相连。
    """
    if not text:
        return text
    out: list[str] = []
    for raw in text.split("\n"):
        cur = raw.rstrip()
        if not out or not cur:
            out.append(cur)
            continue
        prev = out[-1]
        stripped = cur.lstrip()
        if not stripped:
            out.append(cur)
            continue
        # 上一行为空 / 以终止符结尾 → 不合并
        if not prev or prev[-1] in _SENTENCE_END:
            out.append(cur)
            continue
        # 当前行是标题 / 列表 / 表格 / 引用 / 代码 → 不合并
        if stripped.startswith(_NO_MERGE_PREFIX) or _LIST_RE.match(stripped):
            out.append(cur)
            continue
        # 合并：英文单词之间补空格，中文直接相连
        prev_end, cur_start = prev[-1], stripped[0]
        need_space = prev_end.isascii() and prev_end.isalnum() and cur_start.isascii() and cur_start.isalnum()
        out[-1] = prev + (" " if need_space else "") + stripped
    return "\n".join(out)


def find_header_footer(pages: Iterable[str]) -> set[str]:
    """找出跨页重复出现的行（页眉/页脚）。

    判据：只看每页首尾各 3 行（页眉页脚不会出现在正文中部），
    归一化后长度较短且在 >= 阈值比例的页面里出现过。
    """
    pages = [p for p in pages if p]
    if len(pages) < 3:  # 页数太少不足以判定重复性
        return set()
    counter: dict[str, int] = {}
    for page in pages:
        lines = [l.strip() for l in page.split("\n") if l.strip()]
        for line in set(lines[:3] + lines[-3:]):
            if len(line) <= _HEADER_FOOTER_MAX_LEN:
                counter[line] = counter.get(line, 0) + 1
    return {line for line, cnt in counter.items()
            if cnt / len(pages) >= _HEADER_FOOTER_MIN_RATIO}


def strip_header_footer(pages: list[str], noise: set[str] | None = None) -> list[str]:
    """从每页文本中剔除页眉页脚与纯页码行。"""
    if noise is None:
        noise = find_header_footer(pages)
    cleaned = []
    for page in pages:
        lines = [l for l in page.split("\n")
                 if l.strip() not in noise and not _is_numbered_or_pure_number(l)]
        cleaned.append("\n".join(lines))
    return cleaned


def is_trivial(text: str, min_len: int = 2) -> bool:
    """垃圾块判定：过短、纯符号、纯数字/页码。"""
    s = (text or "").strip()
    if not s:
        return True
    if _is_numbered_or_pure_number(s):
        return True
    if len(s) < min_len:
        return True
    # 没有任何中英文或数字 → 视为噪声（如 ------ 、······）
    return not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", s)


def clean_text(text: str) -> str:
    """单段文本清洗：归一化 → 断行合并 → 再归一化。"""
    if not CLEANER_ENABLED or not text:
        return text
    text = normalize(text)
    text = merge_broken_lines(text)
    return normalize(text)


def clean_parsed(parsed_docs: list[dict]) -> list[dict]:
    """清洗 parse_file 的输出（返回新列表，并丢弃整块都是噪声的页）。

    parsed_docs: [{text, page, source, type, image}, ...]
    跨页信息用于识别页眉页脚，所以要整份文档一起处理，不能逐条洗。
    """
    if not CLEANER_ENABLED or not parsed_docs:
        return parsed_docs

    texts = [normalize(d.get("text") or "") for d in parsed_docs]
    noise = find_header_footer(texts)
    if noise:
        texts = strip_header_footer(texts, noise)

    result = []
    for doc, text in zip(parsed_docs, texts):
        text = normalize(merge_broken_lines(text))
        if is_trivial(text):
            continue  # 整块都是噪声，直接丢
        new_doc = dict(doc)
        new_doc["text"] = text
        result.append(new_doc)
    return result
