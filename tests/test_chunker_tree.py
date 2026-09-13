"""标题树分块测试（对标 MaxKB 的 parent_chain 做法）"""
import pytest

from src.chunker import (
    _parse_sections,
    _common_prefix,
    _format_path,
    chunk_text_tree,
    chunk_parsed,
    split_text,
)


# ---------------------------------------------------------------- 章节切分

def test_parse_sections_basic():
    md = "# 一级\n正文A\n## 二级\n正文B"
    secs = _parse_sections(md)
    assert [s["path"] for s in secs] == [["一级"], ["一级", "二级"]]
    assert secs[0]["body"] == "正文A"
    assert secs[1]["body"] == "正文B"


def test_parse_sections_stack_pops_back():
    """# A / ## B / ### C / ## D —— D 是 B 的兄弟（同为二级），路径应为 A > D，不能还挂着 C。"""
    md = "# A\nA正文\n## B\nB正文\n### C\nC正文\n## D\nD正文"
    paths = [s["path"] for s in _parse_sections(md)]
    assert paths == [["A"], ["A", "B"], ["A", "B", "C"], ["A", "D"]]


def test_code_fence_hash_is_not_heading():
    """代码块里的 # 是注释，不能被当成标题。"""
    md = "# 标题\n```python\n# 这是注释\nprint(1)\n```\n正文"
    secs = _parse_sections(md)
    assert len(secs) == 1
    assert "path" in secs[0] and secs[0]["path"] == ["标题"]
    assert "# 这是注释" in secs[0]["body"]


def test_content_before_first_heading():
    md = "开头引言\n# 标题\n正文"
    secs = _parse_sections(md)
    assert secs[0]["path"] == []
    assert secs[0]["body"] == "开头引言"


# ---------------------------------------------------------------- 路径前缀

def test_chunk_has_parent_path_prefix():
    md = "# Agent 基础原理\n\n## ReAct 模式\n循环一直持续，直到模型输出最终答案才停止。"
    chunks = chunk_text_tree(md)
    assert len(chunks) >= 1
    assert chunks[0].startswith("Agent 基础原理 > ReAct 模式\n")
    assert "循环一直持续" in chunks[0]


def test_short_sections_are_merged():
    """相邻短章节应打包进同一块，避免一堆几十字的碎块。"""
    md = "# 大标题\n" + "".join(f"## 小节{i}\n内容{i}\n" for i in range(5))
    chunks = chunk_text_tree(md, chunk_size=500)
    assert len(chunks) == 1
    assert "小节0" in chunks[0] and "小节4" in chunks[0]


def test_oversized_section_is_split_with_prefix_kept():
    """超长章节滑窗再切，每一片都要带上路径前缀。"""
    long_body = "这是一段很长的中文正文。" * 60   # 约 720 字
    md = f"# 章\n## 节\n{long_body}"
    chunks = chunk_text_tree(md, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.startswith("章 > 节\n")


def test_no_heading_falls_back_to_sliding_window():
    """无标题文本（PDF 页 / OCR）必须退回滑窗，行为与旧版一致。"""
    text = "这是一段没有任何标题的纯文本。" * 60
    tree = chunk_text_tree(text, chunk_size=500, overlap=50)
    plain = split_text(text, chunk_size=500, overlap=50)
    assert tree == plain
    assert len(tree) >= 2


def test_common_prefix():
    assert _common_prefix([["A", "B"], ["A", "C"]]) == ["A"]
    assert _common_prefix([["A"], ["B"]]) == []
    assert _common_prefix([["A", "B"], ["A", "B", "C"]]) == ["A", "B"]


def test_format_path_truncates_from_front():
    """路径过长时丢前面的层级，保留最近的上下文。"""
    long_titles = ["超长标题" * 20, "另一超长标题" * 20, "结尾"]
    path = _format_path(long_titles)
    assert path.startswith("…")
    assert path.endswith("结尾")


# ---------------------------------------------------------------- 主入口

def test_chunk_parsed_tree_mode(monkeypatch):
    """开关打开时，chunk 内容带父路径前缀。"""
    monkeypatch.setattr("src.chunker.CHUNK_TREE_ENABLED", True)
    docs = [{"text": "# 标题\n正文内容在这里", "page": 3, "source": "a.md", "type": "text", "image": None}]
    chunks = chunk_parsed(docs)
    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("标题\n")
    assert chunks[0]["page"] == 3
    assert chunks[0]["source"] == "a.md"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["type"] == "text"


def test_chunk_parsed_window_mode_is_untouched(monkeypatch):
    """开关关闭（当前默认）时，走滑动窗口，原文一个字不动、无路径前缀。"""
    monkeypatch.setattr("src.chunker.CHUNK_TREE_ENABLED", False)
    text = "# 标题\n正文内容在这里"
    docs = [{"text": text, "page": None, "source": "a.md", "type": "text", "image": None}]
    chunks = chunk_parsed(docs)
    assert len(chunks) == 1
    assert chunks[0]["content"] == text


def test_chunk_parsed_preserves_image_type():
    """图片/OCR 块必须透传 type 与 image，否则多模态路径会断。"""
    docs = [{"text": "发票金额 100 元", "page": 1, "source": "p.png", "type": "image", "image": "/tmp/p.png"}]
    chunks = chunk_parsed(docs)
    assert chunks[0]["type"] == "image"
    assert chunks[0]["image"] == "/tmp/p.png"


def test_empty_input():
    assert chunk_text_tree("") == []
    assert chunk_text_tree("   \n\n ") == []
    assert chunk_parsed([]) == []
