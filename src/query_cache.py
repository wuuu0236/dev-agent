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
import os
import struct
import sys
from datetime import datetime

import numpy as np

from src.config import (
    DB_PATH, QUERY_CACHE_ENABLED, QUERY_CACHE_THRESHOLD, QUERY_CACHE_MAX_PER_KB,
)

_CACHE_TABLE = "query_cache"

# 缓存向量版本：当"查询侧怎么编码向量"发生变化时（典型如启用 BGE 指令前缀），
# 新旧向量不再在同一个语义空间，旧缓存若继续参与比对就会永不命中甚至错配。
# 用版本号做隔离：旧行自动失配失效，新缓存写入当前版本，无需手动清表。
CACHE_VEC_VERSION = os.getenv("CACHE_VEC_VERSION", "bge_q1")


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
    # 兼容旧库：vec_ver 列可能不存在，补上。旧行默认为空串 → 与新版本号不匹配
    # → 自动失效，不用手动清表。
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(" + _CACHE_TABLE + ")")}
    if "vec_ver" not in cols:
        conn.execute(
            "ALTER TABLE " + _CACHE_TABLE + " ADD COLUMN vec_ver TEXT NOT NULL DEFAULT ''"
        )
    conn.commit()
    conn.close()


def _embed(text: str) -> np.ndarray:
    """问题转向量。惰性 import embeddings：CI/测试无 key 也能 import 本模块。

    走 embed_query（带 BGE 指令前缀），与检索侧保持一致——
    两边编码方式一致，缓存的余弦对比才有意义。
    """
    from src.embeddings import embed_query
    return np.asarray(embed_query(text), dtype=np.float32)


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
        f"SELECT embedding, answer, sources FROM {_CACHE_TABLE}"
        f" WHERE kb_id = ? AND vec_ver = ?",
        (kb_id, CACHE_VEC_VERSION),
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
        f"INSERT INTO {_CACHE_TABLE} (kb_id, question, embedding, answer, sources, created_at, vec_ver)"
        f" VALUES (?, ?, ?, ?, ?, ?, ?)",
        (kb_id, query, _pack(_embed(query)), answer,
         json.dumps(sources, ensure_ascii=False),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), CACHE_VEC_VERSION),
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


def invalidate_cached_answer(kb_id: str, question: str) -> bool:
    """按语义作废**一条**缓存，返回是否真的删到了（反馈环 P2-L1）。

    与 `clear_kb_cache` 的区别只在粒度，但触发场景完全不同：那个是"知识库变了，
    整库缓存全部作废"；这个是"用户说这条答错了，而它恰好是缓存给出的"。

    为什么必须支持单条：👎 一条缓存命中的记录，含义是**这段错的答案被缓存了，
    会持续错下去**——同一个人不会问第二遍，所以它永远不会被自然纠正。但改用
    整库清空的话，会顺带干掉几百条正常缓存（下次全部重算），删一条错答案的
    代价变成一整库的重算，不划算。

    匹配逻辑与 `get_cached_answer` **逐行一致**（同一个 `_embed` / `_cosine` /
    同一个阈值）。这不是巧合：两边但凡有一点不一致，就会出现"存的时候能命中、
    删的时候删不掉"的不对称——而这种 bug 是静默的，用户以为删了，下次照旧吐出
    同一个错答案。

    失败（无 key / 网络抖动）一律返回 False 且不抛：它是旁路动作，绝不能因为
    删不掉缓存，就让"打反馈"这个操作本身失败。
    """
    if not QUERY_CACHE_ENABLED:
        return False
    try:
        _init_table()
        q_vec = _embed(question)
        conn = _connect()
        rows = conn.execute(
            f"SELECT rowid, embedding FROM {_CACHE_TABLE}"
            f" WHERE kb_id = ? AND vec_ver = ?",
            (kb_id, CACHE_VEC_VERSION),
        ).fetchall()
        best_rowid, best_sim = None, QUERY_CACHE_THRESHOLD
        for row in rows:
            sim = _cosine(q_vec, _unpack(row["embedding"]))
            if sim >= best_sim:
                best_sim, best_rowid = sim, row["rowid"]
        if best_rowid is None:
            conn.close()
            return False
        conn.execute(f"DELETE FROM {_CACHE_TABLE} WHERE rowid = ?", (best_rowid,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[QueryCache] 作废单条缓存失败（旁路，不影响反馈本身）: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return False
