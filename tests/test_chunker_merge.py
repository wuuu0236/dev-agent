# -*- coding: utf-8 -*-
"""碎块合并（CHUNK_MIN_SIZE）测试。

背景：CHUNK_MIN_SIZE 这个配置在 config.py 里定义了、也被 chunker import 了，
但一直没有真正参与计算——是个"死配置"。本文件把它的行为固定下来。

注意：split_text / chunk_document 的 chunk_size 是**默认参数**（定义时求值），
monkeypatch CHUNK_SIZE 对它们无效。所以端到端用例按真实默认值（500/50）
构造数据，只有 chunk_parsed 内部引用的 CHUNK_MIN_SIZE 可以被 monkeypatch。
"""
import pytest

from src import chunker
from src.chunker import merge_tiny_chunks, chunk_parsed


def test_tail_tiny_chunk_merges_into_previous():
    """尾块过短 → 并入前一块，顺序不能反。"""
    chunks = ["A" * 300, "B" * 30]
    out = merge_tiny_chunks(chunks, min_size=120)
    assert len(out) == 1
    assert out[0].startswith("A" * 300)
    assert out[0].endswith("B" * 30)


def test_head_tiny_chunk_merges_into_next():
    """首块过短（前面没有块可并）→ 往后并。"""
    chunks = ["A" * 30, "B" * 300]
    out = merge_tiny_chunks(chunks, min_size=120)
    assert len(out) == 1
    assert out[0].startswith("A" * 30)
    assert out[0].endswith("B" * 300)


def test_single_tiny_chunk_is_kept():
    """只有一个块时没有邻居可并——必须原样保留，否则内容直接丢失。"""
    assert merge_tiny_chunks(["很短"], min_size=120) == ["很短"]


def test_all_chunks_long_enough_are_untouched():
    chunks = ["A" * 200, "B" * 200, "C" * 200]
    assert merge_tiny_chunks(chunks, min_size=120) == chunks


def test_disabled_when_min_size_zero():
    """min_size=0 表示关闭合并，行为必须与改动前完全一致。"""
    chunks = ["A" * 300, "B" * 10]
    assert merge_tiny_chunks(chunks, min_size=0) == chunks


def test_disabled_when_min_size_negative():
    chunks = ["A" * 300, "B" * 10]
    assert merge_tiny_chunks(chunks, min_size=-1) == chunks


def test_multiple_consecutive_tiny_chunks_all_merged():
    """连续多个碎块要全部并进前一块，不能只合并第一个。"""
    chunks = ["A" * 300, "b" * 10, "c" * 10, "d" * 10]
    out = merge_tiny_chunks(chunks, min_size=120)
    assert len(out) == 1
    assert out[0].count("\n") == 3


def test_empty_list():
    assert merge_tiny_chunks([], min_size=120) == []


def test_boundary_exactly_min_size_is_not_merged():
    """恰好等于阈值不算过短（判据是 <），边界行为固定住。"""
    chunks = ["A" * 300, "B" * 120]
    assert len(merge_tiny_chunks(chunks, min_size=120)) == 2


def test_merge_uses_newline_separator():
    """片段之间补换行，避免两句直接粘成一个词。"""
    out = merge_tiny_chunks(["A" * 300, "B" * 30], min_size=120)
    assert "\n" in out[0]


# ---------------------------------------------------------------- 端到端


def test_chunk_parsed_merges_sliding_window_tail(monkeypatch):
    """端到端：滑窗步长 450（500-50），1000 字会切成 500/500/100，尾块必须被并掉。"""
    monkeypatch.setattr(chunker, "CHUNK_TREE_ENABLED", False)
    monkeypatch.setattr(chunker, "CHUNK_MIN_SIZE", 120)

    docs = [{"text": "正" * 1000, "page": 1, "source": "a.md"}]
    chunks = chunk_parsed(docs)

    assert len(chunks) == 2                     # 原本 3 块，尾块被合并
    assert all(len(c["content"]) >= 120 for c in chunks)
    assert [c["chunk_index"] for c in chunks] == [0, 1]


def test_chunk_parsed_keeps_tail_when_disabled(monkeypatch):
    """关掉合并（min_size=0）后尾块保持独立，确认开关真的能回到旧行为。"""
    monkeypatch.setattr(chunker, "CHUNK_TREE_ENABLED", False)
    monkeypatch.setattr(chunker, "CHUNK_MIN_SIZE", 0)

    docs = [{"text": "正" * 1000, "page": 1, "source": "a.md"}]
    chunks = chunk_parsed(docs)

    assert len(chunks) == 3
    assert len(chunks[-1]["content"]) == 100


def test_chunk_parsed_merges_short_last_section_in_tree_mode(monkeypatch):
    """端到端：标题树模式下，末尾短章节本会单独成块，必须被并掉。"""
    monkeypatch.setattr(chunker, "CHUNK_TREE_ENABLED", True)
    monkeypatch.setattr(chunker, "CHUNK_MIN_SIZE", 120)

    # 甲章节正文 498 字，加上标题渲染后刚好撑满一块；乙章节只剩一句话
    text = "# 甲\n" + "甲" * 498 + "\n# 乙\n短句"
    docs = [{"text": text, "page": 1, "source": "b.md"}]
    chunks = chunk_parsed(docs)

    assert all(len(c["content"]) >= 120 for c in chunks)


def test_chunk_parsed_preserves_metadata(monkeypatch):
    """合并只能动 content，page / source / type / image 必须原样透传。"""
    monkeypatch.setattr(chunker, "CHUNK_TREE_ENABLED", False)
    monkeypatch.setattr(chunker, "CHUNK_MIN_SIZE", 120)

    docs = [{"text": "正" * 1000, "page": 7, "source": "c.md",
             "type": "image", "image": "c.png"}]
    chunks = chunk_parsed(docs)

    assert all(c["page"] == 7 for c in chunks)
    assert all(c["source"] == "c.md" for c in chunks)
    assert all(c["type"] == "image" for c in chunks)
    assert all(c["image"] == "c.png" for c in chunks)
