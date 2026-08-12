"""
query_cache.py — 语义缓存（RAG 问答路径）

原理：用户提问转成向量，与缓存里存过的问题向量算余弦相似度，
超过阈值就认为"意思一样"，直接返回缓存答案——不检索、不调模型。

本质是"数据库查询缓存的知识库版"（键换成向量、按相似度模糊匹配）：
  · 命中 = 秒回，0 API 成本
  · 失效 = 知识库文档变化（等价于"表被写入"），见 vector_store 的 clear_kb_cache 钩子

只缓存**无历史**的独立提问：带历史的追问是个性化的，命中率低且易错配。

测试约定：本模块顶层不 import src.embeddings（CI 无 key 环境），
_embed 惰性 import，测试用固定向量 monkeypatch。
"""
import json
import struct
from datetime import datetime

import numpy as np

from src.config import (
    DB_PATH, QUERY_CACHE_ENABLED, QUERY_CACHE_THRESHOLD, QUERY_CACHE_MAX_PER_KB,
)

_CACHE_TABLE = "query_cache"


def _connect():
    """SQLite 连接（每次新建，避免多线程共享）。"""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_table():
    # 不用 f-string：SQL 注释里的 [{source, page, type}] 会被当成变量插值
    conn = _connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS " + _CACHE_TABLE + " (\n"
        "    kb_id      TEXT NOT NULL,\n"
        "    question   TEXT NOT NULL,\n"
        "    embedding  BLOB NOT NULL,      -- 问题向量（float32 打包）\n"
        "    answer     TEXT NOT NULL,      -- 完整答案\n"
        "    sources    TEXT NOT NULL,      -- 引用来源 JSON（[{source, page, type}]）\n"
        "    created_at TEXT NOT NULL\n"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_kb ON " + _CACHE_TABLE + "(kb_id)")
    conn.commit()
    conn.close()


def _embed(text: str) -> np.ndarray:
    """问题转向量。惰性 import embeddings：CI/测试无 key 也能 import 本模块。"""
    from src.embeddings import embed_single
    return np.asarray(embed_single(text), dtype=np.float32)


def _pack(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_cached_answer(kb_id: str, query: str) -> dict | None:
    """语义查找缓存。命中返回 {answer, sources}，否则 None。"""
    if not QUERY_CACHE_ENABLED:
        return None
    _init_table()
    q_vec = _embed(query)
    conn = _connect()
    rows = conn.execute(
        f"SELECT embedding, answer, sources FROM {_CACHE_TABLE} WHERE kb_id = ?",
        (kb_id,),
    ).fetchall()
    conn.close()

    best, best_sim = None, QUERY_CACHE_THRESHOLD
    for row in rows:
        sim = _cosine(q_vec, _unpack(row["embedding"]))
        if sim >= best_sim:
            best_sim, best = sim, row
    if best is None:
        return None
    return {"answer": best["answer"], "sources": json.loads(best["sources"])}


def cache_answer(kb_id: str, query: str, answer: str, sources: list[dict]):
    """存入一条缓存。每库超过上限时删最旧的。"""
    if not QUERY_CACHE_ENABLED or not answer:
        return
    _init_table()
    conn = _connect()
    conn.execute(
        f"INSERT INTO {_CACHE_TABLE} (kb_id, question, embedding, answer, sources, created_at)"
        f" VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, query, _pack(_embed(query)), answer,
         json.dumps(sources, ensure_ascii=False),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    # 容量控制：超过上限，删掉最旧（rowid 最小）的多余条目
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM {_CACHE_TABLE} WHERE kb_id = ?", (kb_id,)
    ).fetchone()
    excess = row["c"] - QUERY_CACHE_MAX_PER_KB
    if excess > 0:
        conn.execute(
            f"DELETE FROM {_CACHE_TABLE} WHERE rowid IN ("
            f"  SELECT rowid FROM {_CACHE_TABLE} WHERE kb_id = ?"
            f"  ORDER BY rowid ASC LIMIT ?)",
            (kb_id, excess),
        )
    conn.commit()
    conn.close()


def clear_kb_cache(kb_id: str):
    """清空某知识库的缓存。知识库文档变化（新增/删除）时调用。"""
    if not QUERY_CACHE_ENABLED:
        return
    _init_table()
    conn = _connect()
    conn.execute(f"DELETE FROM {_CACHE_TABLE} WHERE kb_id = ?", (kb_id,))
    conn.commit()
    conn.close()
