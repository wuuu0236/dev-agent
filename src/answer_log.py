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

本模块同时承载反馈环的采集与回流（P1 / P2）：

  · **P1 采集** —— `set_rating()` 写入 👍/👎 与备注（`comment`）。
  · **P2-L1 自动** —— 👎 落在**命中缓存**的记录上时，自动作废那条缓存
    （`_purge_cached_answer` → `query_cache.invalidate_cached_answer`），
    并把 `resolved` 记成 `cache_purged`。这档之所以敢自动，判据是**完全可逆**：
    删错了最坏结果只是下次重算一遍。
  · **P2-L2 人工** —— 👎 进 `kb_gaps` 待补清单；人补完原文、重建索引后标记
    `kb_patched`，或判定无需处理标 `ignored`。`resolved` 的写入点全在这里，
    于是"这条反馈最后被怎么处理了"事后可查——一个采集了却没人消费的反馈，
    等于没采集。

  ⚠️ L2 与设计文档（`notes/feedback-loop-plan.md` §6）有一处偏差，如实记录：
  原方案要求「👎 **且备注提到内容错误/缺失**」才进待补清单。实现改为
  **只要 👎 就进**。理由：判断备注里有没有提到"内容缺失"要靠关键词或 NLP，脆弱
  且会漏——漏掉的代价是信号直接消失（用户已经表达了不满，系统却当没看见）；
  多留一条待办的代价只是人扫一眼。清单里同时显示 `gate_score`，人一眼就能分辨
  这是"该补文档"（分数极低）还是"该调检索"（分数接近阈值）。

设计上的三条自我约束：
  1. **旁路，不阻断**：所有写入失败一律吞掉（`log_answer` 内部全包 try/except），
     日志写得再烂也不能影响用户拿到回答。P2 的自动作废同样如此——删不掉缓存
     **绝不能**让"打反馈"这个动作本身失败。
  2. **不重复检索**：`hits` 直接复用已经算好的 contexts，绝不为了记日志再搜一次。
  3. **淘汰要保护信号**：按时间删最旧会把"被 👎 但还没处理"的记录先删掉——那正是
     这个功能的全部产出。见 `_evict()`。

测试约定：顶层不 import embeddings（CI 无 key 环境），只依赖 config + citations。
`_purge_cached_answer` 单独成函数，就是为了测试能替身它、不真调 embedding。
"""
import json
import sqlite3
import sys
from datetime import datetime

from src.config import (
    DB_PATH, RETRIEVAL_MIN_SCORE,
    ANSWER_LOG_ENABLED, ANSWER_LOG_MAX_PER_KB,
    ANSWER_LOG_MAX_HITS, ANSWER_LOG_HIT_CHARS,
    FEEDBACK_CACHE_PURGE_ENABLED,
)

_TABLE = "answer_log"
_GAP_TABLE = "kb_gaps"

# `resolved` 的合法取值（P2）。白名单放在这里而不是散落在调用点，是为了让
# "这条反馈能处于哪些状态"有唯一的事实来源——否则某天写进去一个拼错的值，
# 界面上会显示一个代码里查不到含义的字符串，而且没人会注意到。
RESOLVED_STATES = ("none", "ignored", "cache_purged", "kb_patched", "case_added")

# 待补清单的三态。与 resolved 是两个字段：gap 是"这件事处理到哪一步"，
# resolved 是"这条反馈最终怎么解决的"。映射只写在这一处。
GAP_STATUSES = ("open", "done", "ignored")
_GAP_TO_RESOLVED = {"done": "kb_patched", "ignored": "ignored", "open": "none"}

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
        "    pipeline_reason TEXT DEFAULT '',    -- 执行计划归因（第八章）：chitchat_minimal 等\n"
        "    created_at      TEXT    NOT NULL\n"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_log_kb ON " + _TABLE + "(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_answer_log_rating ON " + _TABLE + "(rating)")
    # 老库迁移：pipeline_reason 是第八章（Query Execution Plan）新增的列。
    # 用 database._migrate_add_column 的同款 PRAGMA 检查（不走它本体：表由本模块
    # 惰性建，init_db 阶段 answer_log 可能还不存在，在那里迁移会对空表 ALTER 报错）。
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}
    if "pipeline_reason" not in existing:
        conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN pipeline_reason TEXT DEFAULT ''")

    # 待补知识清单（P2-L2）。`answer_log_id` 上建**唯一索引**：幂等交给数据库约束，
    # 而不是靠"先查再插"——后者在两个入口同时打反馈时会插出重复行，而那种重复
    # 只有人翻清单时才会发现。
    conn.execute(
        "CREATE TABLE IF NOT EXISTS " + _GAP_TABLE + " (\n"
        "    id            INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "    kb_id         TEXT NOT NULL,\n"
        "    answer_log_id INTEGER NOT NULL,    -- 指向 answer_log.id\n"
        "    note          TEXT DEFAULT '',     -- 用户备注（可空）\n"
        "    status        TEXT DEFAULT 'open', -- open / done / ignored\n"
        "    created_at    TEXT NOT NULL\n"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_gaps_kb ON " + _GAP_TABLE + "(kb_id)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_gaps_log ON "
                 + _GAP_TABLE + "(answer_log_id)")
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
               hits: list[dict] | str | None = None,
               pipeline_reason: str | None = None) -> int | None:
    """写入一条问答快照，返回新行 id；关闭或失败时返回 None。

    三条自我约束（见模块 docstring）：旁路失败不抛、不重复检索、淘汰保护信号。

    `gate_score` 与 `hit_cache` 的语义边界要说清：缓存命中时**根本没检索**，
    所以 gate_score 是 NULL 而不是 0 —— 0 会被误读成"检索了但一分没得"。

    `pipeline_reason`：执行计划的归因标签（docs 第八章），评估面板据此按管线
    类型（chitchat_minimal / simple_expanded / simple_verified / simple_lean /
    multi_hop_full）筛选对比。
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
            " gate_score, hit_cache, backend, hits, pipeline_reason, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kb_id, question, retrieval_query or question, answer or "",
                1 if grounded else 0,
                gate_score, 1 if hit_cache else 0, backend, hits,
                pipeline_reason or "",
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


def set_rating(log_id: int, rating: str | None,
               comment: str | None = None) -> bool:
    """给某条回答打反馈：'up'（有用）/ 'down'（没用）/ None（撤销）。返回是否命中该条。

    **这是反馈环唯一的采集入口**，同时负责把信号分流到出口（P2）——收到信号却
    没人消费，等于没采集：

      · `down`            → 进 `kb_gaps` 待补清单（人工处理，见 list_gaps）
      · `down` + 缓存命中 → 自动作废那条缓存，`resolved='cache_purged'`（P2-L1）
      · `up` / `None`     → 收回该条的 open 待办（撤回信号，不该留下待办垃圾）
      · 带 `comment`      → 一并落库，清单里显示（帮人判断该补文档还是该调检索）

    带 rating 的记录不会被 `_evict` 的容量淘汰清掉（见该函数：只淘汰 rating IS NULL
    的），所以人工标注过的信号不会意外丢失。

    `comment` 只在传入非 None 时覆盖：前端"只点个赞、没写备注"不该把之前写的
    备注抹掉——那属于静默丢数据。
    """
    if rating not in (None, "up", "down"):
        raise ValueError(f"rating 只能是 None / 'up' / 'down'，收到：{rating!r}")

    _init_table()
    conn = _connect()
    row = conn.execute(
        "SELECT kb_id, question, hit_cache FROM " + _TABLE + " WHERE id = ?", (log_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if comment is None:
        cur = conn.execute("UPDATE " + _TABLE + " SET rating = ?, rated_at = ? WHERE id = ?",
                           (rating, now, log_id))
    else:
        cur = conn.execute(
            "UPDATE " + _TABLE + " SET rating = ?, comment = ?, rated_at = ? WHERE id = ?",
            (rating, comment, now, log_id),
        )
    hit = bool(cur.rowcount)
    kb_id, question, hit_cache = row["kb_id"], row["question"], bool(row["hit_cache"])

    # 分流与评分写在同一个事务里：否则会出现"rating 存了、待办没建"的半截状态，
    # 而人只会发现"我明明点了 👎，清单里却没有"，无从判断是没生效还是漏了。
    if rating == "down":
        conn.execute(
            "INSERT INTO " + _GAP_TABLE + " (kb_id, answer_log_id, note, status, created_at)"
            " VALUES (?, ?, ?, 'open', ?)"
            " ON CONFLICT(answer_log_id) DO UPDATE SET"
            "   note   = CASE WHEN excluded.note = '' THEN note ELSE excluded.note END,"
            "   status = 'open'",
            (kb_id, log_id, comment or "", now),
        )
    else:
        # 撤回信号 = 收回待办。已 done / ignored 的不动：那说明人已经处理过了，
        # 用户改主意不该让处理记录消失。
        conn.execute(
            "DELETE FROM " + _GAP_TABLE + " WHERE answer_log_id = ? AND status = 'open'",
            (log_id,),
        )
    conn.commit()
    conn.close()

    # P2-L1：错的答案被缓存了 → 它不会自己消失（同一个人不会再问第二遍，
    # 所以这条错永远不会被自然纠正）。这是唯一能被自动修掉的一类错。
    # 放在事务提交**之后**：作废缓存要走一次 embedding，网络慢的时候不该让
    # "打反馈"这个动作卡住等它。
    if rating == "down" and hit_cache and FEEDBACK_CACHE_PURGE_ENABLED:
        if _purge_cached_answer(kb_id, question):
            mark_resolved(log_id, "cache_purged")
    return hit


def _purge_cached_answer(kb_id: str, question: str) -> bool:
    """作废那条毒缓存（P2-L1 的实际动作），返回是否删到。

    单独成函数有两个原因：① 惰性 import，本模块顶层不碰 embedding，CI 无 key
    环境照样能 import；② 测试可以替身它——不必真调 embedding 也能验证
    "👎 且缓存命中 → 确实走了作废这条路"。
    """
    try:
        from src.query_cache import invalidate_cached_answer
        return invalidate_cached_answer(kb_id, question)
    except Exception as e:
        print(f"[AnswerLog] 作废缓存失败（不影响反馈本身）: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return False


def mark_resolved(log_id: int, resolved: str) -> bool:
    """标记这条反馈最终怎么处理的（P2）。白名单外的值直接抛错。返回是否命中该条。

    为什么必须显式标记：`resolved` 是"闭环走到哪一步"的唯一凭证。采集了却不
    消费的反馈等于没采集——事后没法区分「还没人看」和「看过、决定不改」，
    于是同一批问题会被重复处理，处理过的问题又会被重复怀疑。
    """
    if resolved not in RESOLVED_STATES:
        raise ValueError(f"resolved 只能是 {RESOLVED_STATES} 之一，收到：{resolved!r}")
    _init_table()
    conn = _connect()
    cur = conn.execute("UPDATE " + _TABLE + " SET resolved = ? WHERE id = ?",
                       (resolved, log_id))
    hit = bool(cur.rowcount)
    conn.commit()
    conn.close()
    return hit


# ---------- 待补知识清单（P2-L2）----------

def list_gaps(kb_id: str | None = None, status: str | None = "open",
              limit: int = 200) -> list[dict]:
    """列出待补知识清单（默认只看 open）。给界面用。

    返回的每条**带上原问题的现场**（question / gate_score / hit_cache）：因为
    "该补文档还是该调检索"要靠 `gate_score` 判断（分界线见 gap_stats），
    只给一个问题标题的话，处理的人还得回头翻日志——多这一步就会没人做。

    `status=None` 表示不筛状态（看全部历史）。
    """
    _init_table()
    where, args = [], []
    if kb_id:
        where.append("g.kb_id = ?")
        args.append(kb_id)
    if status:
        where.append("g.status = ?")
        args.append(status)
    sql = ("SELECT g.id, g.kb_id, g.answer_log_id, g.note, g.status, g.created_at,"
           "       a.question, a.retrieval_query, a.gate_score, a.hit_cache,"
           "       a.rating, a.comment"
           "  FROM " + _GAP_TABLE + " g"
           "  LEFT JOIN " + _TABLE + " a ON a.id = g.answer_log_id")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY g.id DESC LIMIT ?"
    args.append(int(limit))

    conn = _connect()
    rows = conn.execute(sql, tuple(args)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_gaps(kb_id: str | None = None, status: str | None = "open") -> int:
    """待补事项条数（`status=None` 数全部）。界面上的角标用。"""
    _init_table()
    where, args = [], []
    if kb_id:
        where.append("kb_id = ?")
        args.append(kb_id)
    if status:
        where.append("status = ?")
        args.append(status)
    sql = "SELECT COUNT(*) AS c FROM " + _GAP_TABLE
    if where:
        sql += " WHERE " + " AND ".join(where)
    conn = _connect()
    row = conn.execute(sql, tuple(args)).fetchone()
    conn.close()
    return row["c"]


def get_gap(gap_id: int) -> dict | None:
    """按 id 取一条待补事项（含原问题现场），不存在返回 None。

    接口层判权限前必须先知道它属于哪个库——否则任何登录用户都能标记别人的
    待办，直接污染对方的质量信号（同 `/api/feedback` 的理由）。
    """
    _init_table()
    conn = _connect()
    row = conn.execute(
        "SELECT g.id, g.kb_id, g.answer_log_id, g.note, g.status, g.created_at,"
        "       a.question, a.gate_score, a.hit_cache"
        "  FROM " + _GAP_TABLE + " g"
        "  LEFT JOIN " + _TABLE + " a ON a.id = g.answer_log_id"
        " WHERE g.id = ?", (gap_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_gap_status(gap_id: int, status: str) -> bool:
    """处理一条待补事项：done（已解决）/ ignored（确认不用管）/ open（重开）。

    同时更新对应的 `resolved`——两个字段分开写迟早会出现"清单里已完成、日志里
    还显示未处理"的错位，而那种错位没人会主动去核对。
    """
    if status not in GAP_STATUSES:
        raise ValueError(f"status 只能是 {GAP_STATUSES} 之一，收到：{status!r}")
    _init_table()
    conn = _connect()
    cur = conn.execute("UPDATE " + _GAP_TABLE + " SET status = ? WHERE id = ?",
                       (status, gap_id))
    hit = bool(cur.rowcount)
    row = conn.execute("SELECT answer_log_id FROM " + _GAP_TABLE + " WHERE id = ?",
                       (gap_id,)).fetchone()
    conn.commit()
    conn.close()
    if row is not None:
        mark_resolved(row["answer_log_id"], _GAP_TO_RESOLVED[status])
    return hit


def clear_kb_gaps(kb_id: str) -> int:
    """清空某知识库的待补清单，返回删除条数。"""
    _init_table()
    conn = _connect()
    cur = conn.execute("DELETE FROM " + _GAP_TABLE + " WHERE kb_id = ?", (kb_id,))
    n = cur.rowcount or 0
    conn.commit()
    conn.close()
    return n


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
    待补清单（`kb_gaps`）一并清掉——它的 `answer_log_id` 指向的正是这批日志，
    留着会变成点进去看不到任何现场的空壳待办。
    """
    _init_table()
    conn = _connect()
    conn.execute("DELETE FROM " + _GAP_TABLE + " WHERE kb_id = ?", (kb_id,))
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
