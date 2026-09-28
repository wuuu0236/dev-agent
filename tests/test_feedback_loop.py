"""反馈环 P2（回流）单元测试。

P0/P1 解决的是"知道哪里错"，这个文件测的是**"知道之后系统真的会不一样"**：

  ① **按条作废，不是整库清空**
     👎 一条缓存命中 = 这段错答案会被持续吐出（同一个人不会问第二遍，所以它
     永远不会被自然纠正）。整库清空会顺带干掉几百条正常缓存，删一条错的代价
     变成一整库重算。断言必须落到"**剩下的正好是那条无关的**"，而不是只看总数
     ——只看总数的话，删错一条也是通过。
  ② **分流不能半截**
     rating 存了但待办没建，用户只会发现"我点了 👎，清单里却没有"，
     无从判断是没生效还是漏了。
  ③ **失败要诚实**
     作废缓存失败（无 key / 网络抖动）时 `resolved` **不能**标成 `cache_purged`
     ——标了等于谎报"已处理"，而那条毒缓存还在原地。
  ④ **撤销信号 = 收回待办，但不抹掉处理记录**
     用户改主意（👎→👍）不该留下待办垃圾；但人已经标过 done 的，那是处理痕迹。

用临时 SQLite，不碰真库、不调 embedding / LLM。
"""
import sqlite3

import numpy as np
import pytest

import src.answer_log as al
import src.query_cache as qc


# ---------- 固定向量：让"语义匹配"在测试里变成确定性的事 ----------
# 阈值默认 0.90，所以给 1.0 / 0.0 两档就够，不必真算余弦。
_VECS = {
    "混合检索": [1.0, 0.0, 0.0],
    "混合检索怎么做": [1.0, 0.0, 0.0],
    "今天天气怎么样": [0.0, 1.0, 0.0],
    "完全不相干的问题": [0.0, 0.0, 1.0],
}


def _fake_embed(text: str) -> np.ndarray:
    return np.asarray(_VECS.get(text, [0.0, 0.0, 1.0]), dtype=np.float32)


def _make_connect(db_file):
    def fake_connect():
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        return conn
    return fake_connect


def _rows(connect, stmt: str, args: tuple = ()) -> list:
    conn = connect()
    out = list(conn.execute(stmt, args).fetchall())
    conn.close()
    return out


# ---------- 测试环境 ----------

@pytest.fixture
def log_env(tmp_path, monkeypatch):
    """answer_log 指向临时库，两个开关都打开。"""
    connect = _make_connect(tmp_path / "log.db")
    monkeypatch.setattr(al, "_connect", connect)
    monkeypatch.setattr(al, "ANSWER_LOG_ENABLED", True)
    monkeypatch.setattr(al, "FEEDBACK_CACHE_PURGE_ENABLED", True)
    return connect


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """query_cache 指向临时库 + 固定向量（不真调 embedding）。"""
    connect = _make_connect(tmp_path / "cache.db")
    monkeypatch.setattr(qc, "_connect", connect)
    monkeypatch.setattr(qc, "QUERY_CACHE_ENABLED", True)
    monkeypatch.setattr(qc, "_embed", _fake_embed)
    return connect


def _seed_log(**kw) -> int:
    """造一条问答记录，默认是"走了检索且过门控"的常规记录。"""
    question = kw.pop("question", "混合检索怎么做")
    params = {"grounded": True, "gate_score": 0.62}
    params.update(kw)
    return al.log_answer("kb1", question, "答案", **params)


# ================================================================
# 一、P2-L1：按条作废缓存
# ================================================================

def test_purge_removes_only_the_matching_one(cache_env):
    """核心断言：**剩下的正好是那条无关的**，而不是"总数少了一条"。"""
    qc.cache_answer("kb1", "混合检索怎么做", "答案A", [])
    qc.cache_answer("kb1", "今天天气怎么样", "答案B", [])
    assert len(_rows(cache_env, "SELECT * FROM query_cache")) == 2

    assert qc.invalidate_cached_answer("kb1", "混合检索") is True

    left = [r["question"] for r in _rows(cache_env, "SELECT question FROM query_cache")]
    assert left == ["今天天气怎么样"]


def test_purge_only_touches_target_kb(cache_env):
    """别的库即使问题完全一样也不能被删——缓存是按库隔离的。"""
    qc.cache_answer("kb1", "混合检索怎么做", "答案A", [])
    qc.cache_answer("kb2", "混合检索怎么做", "答案B", [])

    assert qc.invalidate_cached_answer("kb1", "混合检索") is True

    assert len(_rows(cache_env, "SELECT * FROM query_cache WHERE kb_id = 'kb2'")) == 1


def test_purge_returns_false_when_nothing_similar(cache_env):
    """没有够相似的条目 → 返回 False，且一条都不能少（不能"反正来了就删一条"）。"""
    qc.cache_answer("kb1", "今天天气怎么样", "答案B", [])

    assert qc.invalidate_cached_answer("kb1", "完全不相干的问题") is False
    assert len(_rows(cache_env, "SELECT * FROM query_cache")) == 1


def test_purge_swallows_embed_failure(cache_env, monkeypatch):
    """下游 embedding 挂了（无 key / 网络抖动）不能抛——它是旁路动作。"""
    qc.cache_answer("kb1", "混合检索怎么做", "答案A", [])

    def boom(_text):
        raise RuntimeError("no api key")

    monkeypatch.setattr(qc, "_embed", boom)
    assert qc.invalidate_cached_answer("kb1", "混合检索") is False
    assert len(_rows(cache_env, "SELECT * FROM query_cache")) == 1


def test_purge_is_noop_when_cache_disabled(cache_env, monkeypatch):
    """缓存关闭时行为与改动前一致：什么都不做。"""
    qc.cache_answer("kb1", "混合检索怎么做", "答案A", [])
    monkeypatch.setattr(qc, "QUERY_CACHE_ENABLED", False)

    assert qc.invalidate_cached_answer("kb1", "混合检索") is False
    assert len(_rows(cache_env, "SELECT * FROM query_cache")) == 1


# ================================================================
# 二、P2-L1 的接线：👎 + 缓存命中 → 自动作废
# ================================================================

def _spy_purge(monkeypatch, result=True, calls=None):
    calls = [] if calls is None else calls
    monkeypatch.setattr(al, "_purge_cached_answer",
                        lambda kb_id, question: (calls.append((kb_id, question)), result)[1])
    return calls


def test_cached_hit_down_purges_and_marks_resolved(log_env, monkeypatch):
    """被 👎 的缓存记录：作废那条缓存 + resolved 记成 cache_purged。"""
    calls = _spy_purge(monkeypatch)
    log_id = al.log_answer("kb1", "混合检索怎么做", "缓存里的错答案",
                           gate_score=None, hit_cache=True)

    assert al.set_rating(log_id, "down") is True

    assert calls == [("kb1", "混合检索怎么做")]
    assert al.get_answer(log_id)["resolved"] == "cache_purged"


def test_non_cache_down_does_not_touch_cache(log_env, monkeypatch):
    """没走缓存的记录，👎 不该去动缓存（那条缓存里是别的问题的答案）。"""
    calls = _spy_purge(monkeypatch)
    log_id = _seed_log()

    al.set_rating(log_id, "down")

    assert calls == []
    assert al.get_answer(log_id)["resolved"] == "none"


def test_up_never_purges(log_env, monkeypatch):
    calls = _spy_purge(monkeypatch)
    log_id = al.log_answer("kb1", "问题", "答案", gate_score=None, hit_cache=True)

    al.set_rating(log_id, "up")

    assert calls == []


def test_purge_failure_does_not_lie_about_resolved(log_env, monkeypatch):
    """作废失败时**不能**标 cache_purged —— 标了就是谎报"已处理"，而毒缓存还在。"""
    _spy_purge(monkeypatch, result=False)
    log_id = al.log_answer("kb1", "问题", "错答案", gate_score=None, hit_cache=True)

    al.set_rating(log_id, "down")

    assert al.get_answer(log_id)["resolved"] == "none"


def test_purge_skipped_when_switch_off(log_env, monkeypatch):
    """开关关闭 = 行为与改动前完全一致（只记反馈，不碰缓存）。"""
    monkeypatch.setattr(al, "FEEDBACK_CACHE_PURGE_ENABLED", False)
    calls = _spy_purge(monkeypatch)
    log_id = al.log_answer("kb1", "问题", "错答案", gate_score=None, hit_cache=True)

    al.set_rating(log_id, "down")

    assert calls == []
    assert al.get_answer(log_id)["resolved"] == "none"


# ================================================================
# 三、P2-L2：待补清单
# ================================================================

def test_down_creates_open_gap_with_scene(log_env):
    """👎 进待办，且带出原问题的现场（判"补文档还是调检索"就靠它）。"""
    log_id = al.log_answer("kb1", "混合检索怎么做", "答案", grounded=False, gate_score=0.03)

    al.set_rating(log_id, "down", "库里没有这块内容")

    gaps = al.list_gaps("kb1", "open")
    assert len(gaps) == 1
    assert gaps[0]["answer_log_id"] == log_id
    assert gaps[0]["question"] == "混合检索怎么做"
    assert gaps[0]["gate_score"] == pytest.approx(0.03)
    assert gaps[0]["note"] == "库里没有这块内容"
    assert al.count_gaps("kb1", "open") == 1


def test_up_does_not_create_gap(log_env):
    al.set_rating(_seed_log(), "up")
    assert al.count_gaps("kb1", "open") == 0


def test_second_down_is_idempotent(log_env):
    """重复点 👎 不能长出两条待办——清单里的重复会让人以为有两个问题要处理。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down")
    al.set_rating(log_id, "down")
    assert al.count_gaps("kb1", "open") == 1


def test_down_keeps_old_note_when_new_one_empty(log_env):
    """不带备注再点一次，不能把之前写的备注抹成空（那属于静默丢数据）。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down", "数字是去年的")

    assert al.list_gaps("kb1", "open")[0]["note"] == "数字是去年的"

    al.set_rating(log_id, "down")

    assert al.list_gaps("kb1", "open")[0]["note"] == "数字是去年的"


def test_up_revokes_open_gap(log_env):
    """撤回信号 = 收回待办，不留垃圾。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down")
    al.set_rating(log_id, "up")
    assert al.count_gaps("kb1", "open") == 0


def test_up_keeps_already_handled_gap(log_env):
    """已处理过的不因用户改主意而消失——那是处理记录，不是待办。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down")
    al.set_gap_status(al.list_gaps("kb1", "open")[0]["id"], "done")

    al.set_rating(log_id, "up")

    assert al.count_gaps("kb1", "done") == 1


def test_comment_not_overwritten_when_rating_without_comment(log_env):
    """只点赞/撤销时不传 comment，不能把已存的备注抹掉。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down", "这里说错了")

    al.set_rating(log_id, "up")

    assert al.get_answer(log_id)["comment"] == "这里说错了"


def test_rating_unknown_gap_id_returns_false(log_env):
    assert al.set_rating(99999, "down") is False


# ================================================================
# 四、处理动作 → resolved 的映射
# ================================================================

@pytest.mark.parametrize("status,expected", [
    ("done", "kb_patched"),   # 补了原文 / 重建了索引
    ("ignored", "ignored"),   # 确认不用管
])
def test_gap_status_writes_resolved(log_env, status, expected):
    log_id = _seed_log()
    al.set_rating(log_id, "down")
    gap_id = al.list_gaps("kb1", "open")[0]["id"]

    assert al.set_gap_status(gap_id, status) is True

    assert al.get_gap(gap_id)["status"] == status
    assert al.get_answer(log_id)["resolved"] == expected


def test_reopen_resets_resolved(log_env):
    """重开要把 resolved 退回 none，否则"待办里挂着、日志里写着已处理"会互相矛盾。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down")
    gap_id = al.list_gaps("kb1", "open")[0]["id"]
    al.set_gap_status(gap_id, "done")

    al.set_gap_status(gap_id, "open")

    assert al.get_answer(log_id)["resolved"] == "none"


def test_gap_status_rejects_unknown_value(log_env):
    with pytest.raises(ValueError):
        al.set_gap_status(1, "finished")


def test_mark_resolved_rejects_unknown_value(log_env):
    with pytest.raises(ValueError):
        al.mark_resolved(1, "whatever")


def test_list_gaps_none_status_returns_all(log_env):
    log_id = _seed_log()
    al.set_rating(log_id, "down")
    al.set_gap_status(al.list_gaps("kb1", "open")[0]["id"], "ignored")

    assert al.count_gaps("kb1", "open") == 0
    assert len(al.list_gaps("kb1", None)) == 1


def test_gap_missing_returns_none(log_env):
    assert al.get_gap(12345) is None


def test_clear_kb_log_also_clears_gaps(log_env):
    """删库连带清待办：留着会变成点进去看不到任何现场的空壳待办。"""
    log_id = _seed_log()
    al.set_rating(log_id, "down")

    al.clear_kb_log("kb1")

    assert al.count_answers("kb1") == 0
    assert al.count_gaps("kb1", None) == 0


def test_gap_eviction_protection_still_holds(log_env, monkeypatch):
    """被 👎 的记录不会被容量淘汰清掉——否则待办会指向一条不存在的日志。"""
    monkeypatch.setattr(al, "ANSWER_LOG_MAX_PER_KB", 1)
    log_id = _seed_log()
    al.set_rating(log_id, "down")

    al.log_answer("kb1", "另一条新问题", "答案")   # 触发淘汰

    assert al.get_answer(log_id) is not None
    assert al.list_gaps("kb1", "open")[0]["answer_log_id"] == log_id
