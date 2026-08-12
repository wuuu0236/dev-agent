"""query_cache 语义缓存单元测试。

用固定向量 monkeypatch 掉 _embed，不调 embedding API：
  · "什么是混合检索" → [1, 0]
  · "混合检索是啥"   → [0.98, 0.02]  （语义近，余弦 ≈ 0.98 > 阈值 0.9）
  · "LangGraph 是什么"→ [0, -1]       （语义远）
  · "什么是向量检索" → [0, 1]
  · "什么是RAG"      → [-1, 1]
"""
import sqlite3

import numpy as np
import pytest

import src.query_cache as qc


@pytest.fixture
def cache(tmp_path):
    """把 query_cache 指向临时库 + 固定向量 embedding。"""
    db_file = tmp_path / "test_cache.db"
    orig_connect = qc._connect

    def fake_connect():
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        return conn

    def fake_embed(text: str) -> np.ndarray:
        if "向量检索" in text:
            return np.array([0.0, 1.0], dtype=np.float32)
        if "RAG" in text:
            return np.array([-1.0, 1.0], dtype=np.float32)
        if "LangGraph" in text:
            return np.array([0.0, -1.0], dtype=np.float32)
        if "是啥" in text:
            return np.array([0.98, 0.02], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)  # 默认："什么是混合检索"

    qc._connect = fake_connect
    qc._embed = fake_embed
    yield qc
    qc._connect = orig_connect


def test_hit_same_question(cache):
    """同问题命中：答案与引用来源都还原。"""
    cache.cache_answer("kb1", "什么是混合检索", "答案A",
                       [{"source": "a.md", "page": 1, "type": "text"}])
    got = cache.get_cached_answer("kb1", "什么是混合检索")
    assert got is not None
    assert got["answer"] == "答案A"
    assert got["sources"] == [{"source": "a.md", "page": 1, "type": "text"}]


def test_hit_similar_question(cache):
    """换说法命中：语义相近算同一问题。"""
    cache.cache_answer("kb1", "什么是混合检索", "答案A", [])
    got = cache.get_cached_answer("kb1", "混合检索是啥")
    assert got is not None and got["answer"] == "答案A"


def test_miss_different_question(cache):
    """语义不同不命中。"""
    cache.cache_answer("kb1", "什么是混合检索", "答案A", [])
    assert cache.get_cached_answer("kb1", "LangGraph 是什么") is None


def test_kb_isolated(cache):
    """不同知识库的缓存互不串。"""
    cache.cache_answer("kb1", "什么是混合检索", "答案A", [])
    assert cache.get_cached_answer("kb2", "什么是混合检索") is None


def test_clear(cache):
    """clear_kb_cache 只清指定库。"""
    cache.cache_answer("kb1", "什么是混合检索", "答案A", [])
    cache.cache_answer("kb2", "什么是混合检索", "答案B", [])
    cache.clear_kb_cache("kb1")
    assert cache.get_cached_answer("kb1", "什么是混合检索") is None
    assert cache.get_cached_answer("kb2", "什么是混合检索") is not None


def test_capacity_limit(cache, monkeypatch):
    """超过上限删最旧（rowid 最小）。"""
    monkeypatch.setattr(qc, "QUERY_CACHE_MAX_PER_KB", 2)
    cache.cache_answer("kb1", "什么是混合检索", "A1", [])
    cache.cache_answer("kb1", "什么是向量检索", "A2", [])
    cache.cache_answer("kb1", "什么是RAG", "A3", [])
    # A1 最早被淘汰
    assert cache.get_cached_answer("kb1", "什么是混合检索") is None
    assert cache.get_cached_answer("kb1", "什么是向量检索") is not None
    assert cache.get_cached_answer("kb1", "什么是RAG") is not None
