"""
answer_log.py — 问答现场快照（反馈环 P0：黑匣子）

为什么需要：RAG 的失败**不可自证**。检索没命中、库里根本没有这段内容，模型不会报错
——它会拿手里的低分上下文编一个读起来很通顺的答案。门控（`answer_gate`）能挡住
"整库都没有"，**挡不住"库里有、只是没检索到"**，而这两种失败在系统内部长得一模一样：
都是一次"成功"的调用，都返回 200，都写了缓存。

所以用户说"这条答错了"时，如果没有现场快照，你根本无从归因——看不到当时检索了什么、
分数多少、门控有没有拦。本模块把这份现场存下来。

**独立价值（不依赖任何用户操作）**：`grounded=0` 的记录天然就是一份
「用户问了、但库里的东西没撑住」的清单，属于 implicit feedback。按 `gate_score`
排序还能分出两类完全不同的待办：

  · `gate_score` 接近阈值 → 差一点，**该调检索**（分块 / 权重 / 阈值）
  · `gate_score` 极低     → 库里连语义邻居都没有，**该补文档**

没有这个字段，两类问题会被混成一句"检索效果不好"，然后去调一个根本没调错的参数。

⚠️ **一条硬约束（决定了抓取点在哪）**：`rag_qa` 的门控判定后会执行 `contexts = []`
——分数在这一行被销毁，返回体里的 contexts 是清空后的。所以快照**必须在门控前抓**，
否则事后永远不知道当时的最高分是 0.28 还是 0.02。

设计上的三条自我约束：
  1. **旁路，不阻断**：所有写入失败一律吞掉（`log_answer` 内部全包 try/except），
     日志写得再烂也不能影响用户拿到回答。
  2. **不重复检索**：`hits` 直接复用已经算好的 contexts，绝不为了记日志再搜一次。
  3. **淘汰要保护信号**：按时间删最旧会把"被 👎 但还没处理"的记录先删掉——那正是
     这个功能的全部产出。见 `_evict()`。

测试约定：顶层不 import embeddings（CI 无 key 环境），只依赖 config + citations。
"""
import json
import sqlite3
import sys
from datetime import datetime

from src.config import (
    DB_PATH, RETRIEVAL_MIN_SCORE,
    ANSWER_LOG_ENABLED, ANSWER_LOG_MAX_PER_KB,
    ANSWER_LOG_MAX_HITS, ANSWER_LOG_HIT_CHARS,
)

_TABLE = "answer_log"

# 缺口分类的分界线（相对阈值，不是绝对值）。
# 为什么用比例而不是写死 0.10：阈值本身是可配的（RETRIEVAL_MIN_SCORE）。
# 写死绝对值的话，一旦把阈值从 0.3 调到 0.5，分界线就会静默失效——分类全乱，
# 而你不会收到任何提示。
#
# ⚠️ 只有**一条**线（0.33 × 阈值 = 默认 0.099），不是两条。曾有一个
# NEAR_MISS_RATIO=0.65 在这里，但分类逻辑其实只用到这一条（中间地带归「差一点」，
# 理由见 gap_stats），那个常量谁也没读——是死配置。已删除，免得界面上写出
# 一个代码里根本不存在的边界。
#
# 0.33 这个值不是拍的，落在实测分布的空档里（见 answer_gate 的实测记录）：
#   · 库里有答案：最低 0.578（≈ 1.9 × 阈值）
#   · 库里没有 / 不该问库：0.001 ~ 0.125（"你好" 0.125、"1+1" 0.091、
#     "帮我写首诗" 0.010、"今天天气" 0.001）
# 线取 0.099，正好把「0.00x 那一片」和「0.12x 及以上」分开：前者是压根没沾到边
# （该补文档），后者多少沾了点边（可能只是没检索好）。
#
# 已知局限（诚实标注）：0.12 左右的"你好""你能做什么"这类**本就不该问库**的问题
# 会落进「差一点」，被读成"该调检索"。它们在清单里排最后（按分降序），
# 目前靠人一眼看出；要做干净得靠意图路由，那不在 P0 范围。
NO_NEIGHBOR_RATIO = 0.33   # gate_score < 阈值 * 该比例 → 「连语义邻居都没有」


def _connect():
    """SQLite 连接（每次新建，避免多线程共享）。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_table():
    """建表 + 索引。`CREATE TABLE IF NOT EXISTS`，兼容已存在的旧库，不做破坏性迁移。"""
    conn = _connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS " + _TABLE + " (\n"
        "    id              INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "    kb_id           TEXT    NOT NULL,\n"
        "    question        TEXT    NOT NULL,   -- 用户原话\n"
        "    retrieval_query TEXT    NOT NULL,   -- 实际拿去检索的 query（追问消解后）\n"
        "    answer          TEXT    NOT NULL,   -- 最终答案\n"
        "    grounded        INTEGER NOT NULL,   -- 是否通过质量门控（1/0）\n"
        "    gate_score      REAL,               -- 门控判定的最高精排分；缓存命中为 NULL\n"
        "    hit_cache       INTEGER DEFAULT 0,  -- 是否命中语义缓存\n"
        "    backend         TEXT,               -- cloud / ollama\n"
        "    hits            TEXT,               -- 命中快照 JSON（复盘的全部依据）\n"
        "    rating          TEXT,               -- P1：NULL / 'up' / 'down'\n"
        "    comment         TEXT,               -- P1：用户补充说明\n"
        "    rated_at        TEXT,               -- P1\n"
        "    resolved        TEXT DEFAULT 'none',-- P2：none/ignored/cache_purged/kb_patched/case_added\n"
        "    created_at      TEXT    NOT NULL\n"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_log_kb ON " + _TABLE + "(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_log_rating ON " + _TABLE + "(rating)")
    conn.commit()
    conn.close()


def snapshot_hits(contexts: list[dict] | None,
                  max_hits: int | None = None,
                  chars: int | None = None) -> str:
    """把候选快照压成 JSON 字符串（存进 `hits` 列）。

    存的是**检索侧真实字段**，不是加工过的展示数据——复盘时要能看出
    "这条为什么被搜出来"（rerank_score / rrf_score 两个分数都留）。
    """
    from src.citations import snippet  # 与引用来源共用同一套截断逻辑

    limit = ANSWER_LOG_MAX_HITS if max_hits is None else max_hits
    width = ANSWER_LOG_HIT_CHARS if chars is None else chars
    out = []
    for c in (contexts or [])[:limit]:
        out.append({
            "source": c.get("source"),
            "page": c.get("page"),
            "type": c.get("type", "text"),
            "rerank_score": c.get("rerank_score"),
            "rrf_score": c.get("rrf_score"),
            "snippet": snippet(c.get("content", ""), width),
        })
    return json.dumps(out, ensure_ascii=False)


def log_answer(kb_id: str, question: str, answer: str, *,
               retrieval_query: str | None = None,
               grounded: bool = True,
               gate_score: float | None = None,
               hit_cache: bool = False,
               backend: str | None = None,
               hits: list[dict] | str | None = None) -> int | None:
    """写入一条问答快照，返回新行 id；关闭或失败时返回 None。

    三条自我约束（见模块 docstring）：旁路失败不抛、不重复检索、淘汰保护信号。

    `gate_score` 与 `hit_cache` 的语义边界要说清：缓存命中时**根本没检索**，
    所以 gate_score 是 NULL 而不是 0 —— 0 会被误读成"检索了但一分没得"。
    """
    if not ANSWER_LOG_ENABLED:
        return None
    try:
        _init_table()
        if isinstance(hits, list):
            hits = json.dumps(hits, ensure_ascii=False)
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO " + _TABLE + " (kb_id, question, retrieval_query, answer, grounded,"
            " gate_score, hit_cache, backend, hits, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kb_id, question, retrieval_query or question, answer or "",
                1 if grounded else 0,
                gate_score, 1 if hit_cache else 0, backend, hits,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        new_id = cur.lastrowid
        _evict(conn, kb_id)
        conn.commit()
        conn.close()
        return new_id
    except Exception as e:
        # 日志是旁路，不是主链路。写不进去也不能让用户看不到回答。
        print(f"[AnswerLog] 写入失败（不影响回答）: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return None


def _evict(conn, kb_id: str) -> int:
    """容量控制：某库超过上限时淘汰，返回淘汰条数。

    ⚠️ **不能简单地"删最旧"**：被 👎 但还没处理的记录正是这个功能的全部产出，
    按时间淘汰会把最该看的信号先删掉。所以只淘汰「没人评价过、且处理状态为
    初始 / 已忽略」的行。被 👎 的、正在待办的，一律保留——宁可超出上限。
    """
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM " + _TABLE + " WHERE kb_id = ?", (kb_id,)
    ).fetchone()
    excess = row["c"] - ANSWER_LOG_MAX_PER_KB
    if excess <= 0:
        return 0
    cur = conn.execute(
        "DELETE FROM " + _TABLE + " WHERE id IN ("
        "  SELECT id FROM " + _TABLE + " WHERE kb_id = ?"
        "    AND rating IS NULL"
        "    AND resolved IN ('none', 'ignored')"
        "  ORDER BY id ASC LIMIT ?)",
        (kb_id, excess),
    )
    return cur.rowcount or 0


def get_answer(log_id: int) -> dict | None:
    """按 id 取一条（含 hits 解析后的列表）。"""
    _init_table()
    conn = _connect()
    row = conn.execute("SELECT * FROM " + _TABLE + " WHERE id = ?", (log_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def count_answers(kb_id: str | None = None) -> int:
    _init_table()
    conn = _connect()
    if kb_id:
        row = conn.execute("SELECT COUNT(*) AS c FROM " + _TABLE + " WHERE kb_id = ?",
                           (kb_id,)).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) AS c FROM " + _TABLE).fetchone()
    conn.close()
    return row["c"]


def list_answers(kb_id: str | None = None, limit: int = 100, *,
                 only_ungrounded: bool = False,
                 only_rated: bool = False,
                 order: str = "id DESC") -> list[dict]:
    """列出问答记录（默认最新在前）。给评估面板用。

    `order` 只接受白名单内的两种值，避免把外部字符串拼进 SQL。
    """
    _init_table()
    where, args = [], []
    if kb_id:
        where.append("kb_id = ?")
        args.append(kb_id)
    if only_ungrounded:
        where.append("grounded = 0")
    if only_rated:
        where.append("rating IS NOT NULL")
    sql = "SELECT * FROM " + _TABLE
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id ASC" if order == "id ASC" else " ORDER BY id DESC"
    sql += " LIMIT ?"
    args.append(int(limit))

    conn = _connect()
    rows = conn.execute(sql, tuple(args)).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def gap_stats(kb_id: str | None = None) -> dict:
    """反馈环 P0 的核心产出：把「没答上」的记录分成可执行的待办。

    返回：
      total / grounded / ungrounded / cache_hits
      near_miss     —— 差点过阈值：该调检索，**不用补文档**
      no_neighbor   —— 连语义邻居都没有：该补文档
      unreachable   —— 没有分数可比，不参与分类

    分界线只有一条（`NO_NEIGHBOR_RATIO × RETRIEVAL_MIN_SCORE`，默认 0.099），
    边界为什么取这个值见模块顶部。
    """
    _init_table()
    conn = _connect()
    args: tuple = ()
    sql = "SELECT * FROM " + _TABLE
    if kb_id:
        sql += " WHERE kb_id = ?"
        args = (kb_id,)
    rows = conn.execute(sql, args).fetchall()
    conn.close()

    none_line = RETRIEVAL_MIN_SCORE * NO_NEIGHBOR_RATIO

    stats = {
        "total": len(rows), "grounded": 0, "ungrounded": 0, "cache_hits": 0,
        "near_miss": [], "no_neighbor": [], "unreachable": [],
    }
    for r in rows:
        if r["hit_cache"]:
            stats["cache_hits"] += 1
        if r["grounded"]:
            stats["grounded"] += 1
            continue
        stats["ungrounded"] += 1
        score = r["gate_score"]
        item = {"id": r["id"], "question": r["question"],
                "gate_score": score, "created_at": r["created_at"]}
        if score is None:
            # 没走检索（缓存命中）—— 没有分数可比，别硬归类
            stats["unreachable"].append(item)
        elif score < none_line:
            stats["no_neighbor"].append(item)
        else:
            # 剩下的（含"刚过线"到"就差一点"的整个区间）一律归「差点过阈」。
            # 取舍理由：这条线以上无法断定"库里真没有"，而两类待办的成本不对称
            # ——调检索是纯计算、可反复试；补文档要人去找资料。不确定时选更轻的
            # 那个，避免把一个可能检索修好的问题误判成"缺文档"而白补一堆资料。
            # 列表按分数降序（下面 sort），越接近阈值的越靠前，先看最可能修好的。
            stats["near_miss"].append(item)
    # 差点过的排前面（更接近修好），缺内容的按时间倒序
    stats["near_miss"].sort(key=lambda x: (x["gate_score"] is None, -(x["gate_score"] or 0)))
    stats["no_neighbor"].sort(key=lambda x: x["id"], reverse=True)
    return stats


def clear_kb_log(kb_id: str) -> int:
    """清空某知识库的问答日志，返回删除条数。

    与「删文档三处同删」同源：知识库没了，引用它的日志就是孤儿数据。
    """
    _init_table()
    conn = _connect()
    cur = conn.execute("DELETE FROM " + _TABLE + " WHERE kb_id = ?", (kb_id,))
    n = cur.rowcount or 0
    conn.commit()
    conn.close()
    return n


def _row_to_dict(row) -> dict:
    d = dict(row)
    try:
        d["hits"] = json.loads(d.get("hits") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["hits"] = []
    d["grounded"] = bool(d.get("grounded"))
    d["hit_cache"] = bool(d.get("hit_cache"))
    return d
