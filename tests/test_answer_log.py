"""answer_log（反馈环 P0 黑匣子）单元测试 + 抓取点回归测试。

这个模块的价值全在**现场**两个字上：事后拿到"这条答错了"，能不能还原当时
检索了什么、最高分多少、门控有没有拦。所以测试重点不是"能写能读"，而是三条
容易在重构中被无声破坏的约束：

  ① **抓取点在门控前**（最重要）
     `rag_qa` 判定未过门控后会执行 `contexts = []` —— 分数在这一行被销毁。
     如果哪天有人把快照挪到后面"顺手一点"，分数就永远是空的，而**测试不写
     就没人会发现**：功能看着还在，日志也在写，只是每条都没分数。
  ② **淘汰保护信号**
     被 👎 的记录正是这个功能的全部产出，按时间淘汰会先把它们删掉。
  ③ **旁路不阻断**
     日志写失败（磁盘满、表被锁）时用户必须照常拿到回答。

用临时 SQLite 库，不碰真库；不调 embedding / LLM。
"""
import importlib.machinery
import importlib.util
import sqlite3
import sys
import types

import pytest

import src.answer_log as al


# ---------- 测试环境 ----------

class _LogEnv:
    """指向临时库的 answer_log 环境，附一个直连口子用于构造前置数据。

    需要直连是因为「被评价过的记录」现在还没有公开写入口（👍/👎 是 P1），
    测试只能自己 UPDATE 出这个状态。
    """

    def __init__(self, module, connect):
        self.mod = module
        self._connect = connect

    def sql(self, stmt: str, args: tuple = ()) -> None:
        conn = self._connect()
        conn.execute(stmt, args)
        conn.commit()
        conn.close()

    def rows(self, stmt: str, args: tuple = ()) -> list:
        conn = self._connect()
        out = list(conn.execute(stmt, args).fetchall())
        conn.close()
        return out


@pytest.fixture
def log(tmp_path, monkeypatch):
    """把 answer_log 指向临时库，并确保功能开关打开。"""
    db_file = tmp_path / "test_log.db"

    def fake_connect():
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(al, "_connect", fake_connect)
    monkeypatch.setattr(al, "ANSWER_LOG_ENABLED", True)
    return _LogEnv(al, fake_connect)


# ---------- 写 / 读 ----------

def test_write_and_read_back(log):
    """基本往返：字段齐全，hits 从 JSON 还原成列表。"""
    log_id = al.log_answer(
        "kb1", "混合检索怎么做的", "先 BM25 再向量融合",
        retrieval_query="混合检索的实现方式", grounded=True, gate_score=0.62,
        backend="cloud",
        hits=[{"source": "a.md", "page": 2, "rerank_score": 0.62}],
    )
    assert log_id is not None

    rec = al.get_answer(log_id)
    assert rec["kb_id"] == "kb1"
    assert rec["question"] == "混合检索怎么做的"
    assert rec["retrieval_query"] == "混合检索的实现方式"  # 与用户原话分开存
    assert rec["grounded"] is True
    assert rec["gate_score"] == pytest.approx(0.62)
    assert rec["hit_cache"] is False
    assert rec["hits"] == [{"source": "a.md", "page": 2, "rerank_score": 0.62}]


def test_retrieval_query_defaults_to_question(log):
    """单轮提问没改写时，retrieval_query 等于原话（不是空字符串）。"""
    log_id = al.log_answer("kb1", "单轮问题", "答案")
    assert al.get_answer(log_id)["retrieval_query"] == "单轮问题"


def test_gate_score_none_is_not_zero(log):
    """缓存命中记 None（没检索），不是 0（检索了没得分）。

    这两个值在报表里看起来都是"没分"，但含义相反：None 说明这次压根没走检索，
    0 说明走了但一分没得。混起来会让"未命中"统计虚高。
    """
    log_id = al.log_answer("kb1", "命中缓存的问题", "缓存里的答案",
                           gate_score=None, hit_cache=True)
    rec = al.get_answer(log_id)
    assert rec["gate_score"] is None
    assert rec["hit_cache"] is True


def test_disabled_flag_writes_nothing(log, monkeypatch):
    """关掉开关 = 行为与加日志之前完全一致（不写库、返回 None）。"""
    monkeypatch.setattr(al, "ANSWER_LOG_ENABLED", False)
    assert al.log_answer("kb1", "q", "a") is None
    assert al.count_answers() == 0


def test_write_failure_is_swallowed(log, monkeypatch):
    """写失败不能抛给调用方——日志是旁路，不是主链路。"""
    def boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(al, "_connect", boom)
    assert al.log_answer("kb1", "q", "a") is None  # 不抛异常


# ---------- 命中快照 ----------

def test_snapshot_keeps_both_scores_and_truncates(log, monkeypatch):
    """快照要留检索侧的真实字段（两个分数都留），文本按宽度截断。"""
    monkeypatch.setattr(al, "ANSWER_LOG_MAX_HITS", 2)
    monkeypatch.setattr(al, "ANSWER_LOG_HIT_CHARS", 10)

    snap = al.snapshot_hits([
        {"source": "a.md", "page": 1, "content": "字" * 50,
         "rerank_score": 0.62, "rrf_score": 0.031},
        {"source": "b.md", "page": 2, "content": "短", "rerank_score": 0.5},
        {"source": "c.md", "page": 3, "content": "不该出现", "rerank_score": 0.4},
    ])
    import json
    hits = json.loads(snap)
    assert len(hits) == 2, "超过 ANSWER_LOG_MAX_HITS 的命中不该进快照（控制体积）"
    assert hits[0]["rerank_score"] == pytest.approx(0.62)
    assert hits[0]["rrf_score"] == pytest.approx(0.031)
    assert len(hits[0]["snippet"]) <= 11, "长文本必须截断"
    assert hits[1]["snippet"] == "短"


def test_snapshot_handles_empty():
    """检索为空时不炸，返回空数组的 JSON。"""
    assert al.snapshot_hits(None) == "[]"
    assert al.snapshot_hits([]) == "[]"


# ---------- 淘汰 ----------

def test_eviction_protects_rated_rows(log, monkeypatch):
    """⚠️ 核心约束：被评价过的记录绝不淘汰。

    被 👎 但还没处理的记录正是这个功能的产出，按时间淘汰会先删掉最该看的那条。
    """
    monkeypatch.setattr(al, "ANSWER_LOG_MAX_PER_KB", 3)
    ids = [al.log_answer("kb1", f"q{i}", "a") for i in range(1, 4)]
    log.sql("UPDATE answer_log SET rating = 'down' WHERE id = ?", (ids[0],))

    al.log_answer("kb1", "q4", "a")
    al.log_answer("kb1", "q5", "a")

    kept = [r["id"] for r in log.rows("SELECT id FROM answer_log ORDER BY id")]
    assert ids[0] in kept, "被 👎 的记录被淘汰掉了 —— 信号丢了，这个功能就白做了"
    assert len(kept) == 3
    assert kept == [ids[0], ids[2] + 1, ids[2] + 2]  # 1 保留，2、3 被淘汰


def test_eviction_skips_already_handled_rows(log, monkeypatch):
    """已处理（resolved != none/ignored）的记录同样不淘汰——它还要留着做对比。"""
    monkeypatch.setattr(al, "ANSWER_LOG_MAX_PER_KB", 2)
    a = al.log_answer("kb1", "q1", "a")
    log.sql("UPDATE answer_log SET resolved = 'kb_patched' WHERE id = ?", (a,))
    al.log_answer("kb1", "q2", "a")
    al.log_answer("kb1", "q3", "a")
    assert a in [r["id"] for r in log.rows("SELECT id FROM answer_log")]


def test_eviction_is_per_kb(log, monkeypatch):
    """配额按库算：一个库写爆不该把另一个库的记录挤掉。"""
    monkeypatch.setattr(al, "ANSWER_LOG_MAX_PER_KB", 2)
    other = al.log_answer("kb2", "重要问题", "答案")
    for i in range(5):
        al.log_answer("kb1", f"q{i}", "a")
    assert other in [r["id"] for r in log.rows("SELECT id FROM answer_log WHERE kb_id='kb2'")]


# ---------- 缺口分类（P0 的核心产出）----------

def test_gap_stats_splits_into_actionable_buckets(log):
    """三档分类：差点过阈 / 没有语义邻居 / 无法分类。"""
    al.log_answer("kb1", "有答案的", "a", grounded=True, gate_score=0.9)
    near = al.log_answer("kb1", "差一点", "a", grounded=False, gate_score=0.25)
    mid = al.log_answer("kb1", "中间地带", "a", grounded=False, gate_score=0.15)
    far = al.log_answer("kb1", "库里没有", "a", grounded=False, gate_score=0.05)
    cache = al.log_answer("kb1", "缓存命中", "a", grounded=False, gate_score=None,
                          hit_cache=True)

    g = al.gap_stats("kb1")
    assert (g["total"], g["grounded"], g["ungrounded"], g["cache_hits"]) == (5, 1, 4, 1)

    near_ids = [it["id"] for it in g["near_miss"]]
    assert near_ids == [near, mid], "中间地带归「差点过阈」（成本不对称，见模块注释）"
    assert [it["id"] for it in g["no_neighbor"]] == [far]
    assert [it["id"] for it in g["unreachable"]] == [cache]

    # 差点过的排在前面（更接近修好）
    assert g["near_miss"][0]["gate_score"] == pytest.approx(0.25)


def test_empty_retrieval_is_not_unreachable(log):
    """检索结果为空记的是 0.0（真检索了、一条没捞到）→ 归 no_neighbor。

    只有缓存命中的 NULL 才进 unreachable。若哪天把空检索也记成 NULL，
    "库里没有"和"这次没检索"会被并成一类，缺口清单就再也分不出该补文档还是
    该查缓存——而它们是完全不同的两件事。
    """
    empty = al.log_answer("kb1", "空检索", "a", grounded=False, gate_score=0.0)
    cache = al.log_answer("kb1", "缓存命中", "a", grounded=False, gate_score=None,
                          hit_cache=True)
    g = al.gap_stats("kb1")
    assert [it["id"] for it in g["no_neighbor"]] == [empty]
    assert [it["id"] for it in g["unreachable"]] == [cache]


def test_gap_buckets_follow_the_threshold(log, monkeypatch):
    """分档用的是**相对阈值**，不是写死的 0.195/0.099。

    阈值是可配的（RETRIEVAL_MIN_SCORE）。写死绝对值的话，把阈值从 0.3 调到 1.0
    后分界线会静默失效——分类全乱，而不会有人收到提示。
    """
    al.log_answer("kb1", "0.25 分", "a", grounded=False, gate_score=0.25)
    assert len(al.gap_stats("kb1")["near_miss"]) == 1, "按 0.3 阈值，0.25 是差点过阈"

    monkeypatch.setattr(al, "RETRIEVAL_MIN_SCORE", 1.0)
    g = al.gap_stats("kb1")
    assert len(g["near_miss"]) == 0
    assert len(g["no_neighbor"]) == 1, "阈值一旦抬高，同一个 0.25 就该归到「缺文档」"


def test_gap_stats_isolated_per_kb(log):
    al.log_answer("kb1", "q1", "a", grounded=False, gate_score=0.05)
    al.log_answer("kb2", "q2", "a", grounded=False, gate_score=0.05)
    assert al.gap_stats("kb1")["ungrounded"] == 1


# ---------- 清空 / 删库串联 ----------

def test_clear_kb_log_is_scoped(log):
    al.log_answer("kb1", "q1", "a")
    al.log_answer("kb2", "q2", "a")
    assert al.clear_kb_log("kb1") == 1
    assert al.count_answers("kb1") == 0
    assert al.count_answers("kb2") == 1


def test_delete_kb_clears_answer_log(tmp_path, monkeypatch):
    """删库必须连带清日志——否则评估面板会统计到一堆指向已删库的孤儿记录。"""
    import src.database as db

    db_file = tmp_path / "test_main.db"
    monkeypatch.setattr(db, "DB_PATH", db_file)
    monkeypatch.setattr(al, "_connect", lambda: _conn_to(db_file))
    db.init_db()

    kb = db.create_kb("临时库")
    other = db.create_kb("留下的库")
    al.log_answer(kb["id"], "q1", "a")
    al.log_answer(other["id"], "q2", "a")

    db.delete_kb(kb["id"])

    assert al.count_answers(kb["id"]) == 0
    assert al.count_answers(other["id"]) == 1, "不该误删别的库"


def _conn_to(db_file):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    return conn


# ---------- 端到端：抓取点必须在门控之前 ----------

def _ensure_langfuse() -> None:
    """缺 langfuse 时注入一个最小替身（只提供 @observe 直通）。

    为什么不像 test_smoke_imports 那样直接 import：本文件要守的是**主链路行为**
    回归（快照抓取点、旁路不阻断），属于任何环境都该跑得起来的一类。
    为缺一个观测 SDK 而整文件跳过，等于这两条约束没人守。
    ⚠️ 只在真的缺 langfuse 时注入 —— 装了就用真的，避免替身掩盖真问题。
    ⚠️ 判"在不在"必须先看 `sys.modules`，不能用 `find_spec`：替身进过 sys.modules
    之后再调 find_spec 会因为 `__spec__ is None` 抛 ValueError（而不是返回 None），
    第二个用到它的测试就崩了。
    """
    if "langfuse" in sys.modules:
        return
    try:
        import langfuse  # noqa: F401  装了就用真的
        return
    except ImportError:
        pass

    def observe(*a, **kw):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    pkg = types.ModuleType("langfuse")
    dec = types.ModuleType("langfuse.decorators")
    dec.observe = observe
    pkg.decorators = dec
    # 补上 __spec__：任何第三方代码用 find_spec 探测时不会因为它是"裸模块"而报错
    pkg.__spec__ = importlib.machinery.ModuleSpec("langfuse", None)
    dec.__spec__ = importlib.machinery.ModuleSpec("langfuse.decorators", None)
    sys.modules["langfuse"] = pkg
    sys.modules["langfuse.decorators"] = dec


# 最高 0.28 < 默认阈值 0.3 → 必然被门控拦下，且分数是"差一点"那一档（>0）
_LOW_CONTEXTS = [
    {"source": "a.md", "page": 1, "type": "text", "content": "看起来相关但不够",
     "rerank_score": 0.28, "rrf_score": 0.016},
    {"source": "b.md", "page": 4, "type": "text", "content": "更不相关的",
     "rerank_score": 0.05, "rrf_score": 0.014},
]


@pytest.fixture
def rag(log, monkeypatch):
    """rag_qa 主链路，检索与生成都换成假的（不碰 API / 不碰真库）。"""
    _ensure_langfuse()
    import src.rag_qa as rag_qa
    import src.query_cache as qc

    class FakeRetriever:
        def __init__(self, kb_id):
            self.kb_id = kb_id

        def search(self, query, top_k=None):
            return [dict(c) for c in _LOW_CONTEXTS]

    monkeypatch.setattr(rag_qa, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(rag_qa, "stream_generate_answer",
                        lambda *a, **kw: iter(["我不确定", "，知识库里没有相关内容。"]))
    monkeypatch.setattr(rag_qa, "generate_answer",
                        lambda *a, **kw: "我不确定，知识库里没有相关内容。")
    # 缓存关掉：命中分支会抢先写日志，混淆本测试的观察点
    monkeypatch.setattr(qc, "get_cached_answer", lambda *a, **k: None)
    monkeypatch.setattr(qc, "cache_answer", lambda *a, **k: None)
    return rag_qa


def test_gate_score_recorded_even_though_gate_cleared_contexts(rag):
    """**回归测试（本文件存在的首要理由）**：分数必须活过门控。

    门控判定未通过后 `rag_qa` 会执行 `contexts = []`，分数在那行被销毁。
    所以快照只能在门控前抓。若有人把抓取点挪到后面，功能看起来照常工作
    ——日志还在写、记录也有——但每条的 gate_score 都是空的，
    「0.28 该调检索」和「0.02 该补文档」重新变得无从区分。
    """
    gen, sources, contexts, retrieval_query, log_ref = rag.stream_rag_query(
        "kb1", "库里没有的东西", top_k=5)

    assert "".join(gen).endswith("知识库里没有相关内容。")
    assert contexts == [] and sources == [], "门控没拦下参数就没意义了"
    assert log_ref["gate_score"] == pytest.approx(0.28), "分数被门控销毁了"

    rec = al.get_answer(log_ref["id"])
    assert rec["grounded"] is False
    assert rec["gate_score"] == pytest.approx(0.28) and rec["gate_score"] > 0
    assert len(rec["hits"]) == 2, "命中快照也必须在门控前抓，否则事后无从复盘"
    assert rec["hits"][0]["source"] == "a.md"


def test_rag_query_nonstream_also_records_score(rag):
    """非流式路径（评估面板在用）同样守住这条约束。"""
    res = rag.rag_query("kb1", "库里没有的东西", top_k=5)
    assert res["grounded"] is False
    assert res["log_id"] is not None
    assert al.get_answer(res["log_id"])["gate_score"] == pytest.approx(0.28)


def test_log_failure_does_not_break_the_answer(rag, monkeypatch):
    """日志写挂（磁盘满 / 库被锁）时，用户必须照常拿到回答。

    旁路模块最典型的翻车方式：一个非关键的写入失败，把主功能整个带下去。
    """
    def boom(**kw):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(al, "log_answer", boom)

    gen, sources, contexts, retrieval_query, log_ref = rag.stream_rag_query(
        "kb1", "库里没有的东西", top_k=5)
    answer = "".join(gen)

    assert answer.endswith("知识库里没有相关内容。"), "日志失败把回答也带走了"
    assert log_ref["id"] is None


def test_cache_hit_branch_logs_without_score(rag, monkeypatch):
    """缓存命中分支：既要有日志（错答案被缓存会持续错下去），又不能记 0 分。

    缓存命中根本没走检索，记 0 会被读成"检索了但一分没得"，把「未命中」统计灌水。
    """
    import src.query_cache as qc
    monkeypatch.setattr(qc, "get_cached_answer", lambda *a, **k: {
        "answer": "缓存里的旧答案",
        "sources": [{"source": "a.md", "page": 1, "type": "text", "snippets": ["原文"]}],
    })

    gen, sources, contexts, retrieval_query, log_ref = rag.stream_rag_query("kb1", "老问题")
    assert "".join(gen) == "缓存里的旧答案"

    rec = al.get_answer(log_ref["id"])
    assert rec["hit_cache"] is True
    assert rec["gate_score"] is None
    assert rec["grounded"] is True, "带引用的缓存回答 = 当时过了门控（二者等价）"
    assert contexts == [], "缓存分支不返回上下文（缓存里只有来源，没有上下文）"
