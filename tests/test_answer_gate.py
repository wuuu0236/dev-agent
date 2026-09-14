"""answer_gate（检索质量门控 / 阈值拒答）单元测试。

覆盖三块：
  · top_score   —— 精排优先、粗排兜底、脏数据不崩
  · is_grounded —— 阈值语义（>= 通过 / <=0 关闭 / 显式覆盖配置）
  · 真实分布    —— 用 2026-09-14 在 kb 83cd2d0c 上实测的分数当夹具，
                   确认默认阈值 0.3 在真实数据上确实分得开
"""
import pytest

import src.answer_gate as ag


def ctx(score, *, key="rerank_score", source="a.md"):
    """构造一条最小可用的检索结果。"""
    return {"content": "内容", "source": source, "page": 0, "type": "text", key: score}


# ---------- top_score ----------

def test_top_score_empty():
    assert ag.top_score([]) == 0.0


def test_top_score_none():
    assert ag.top_score(None) == 0.0


def test_top_score_takes_max():
    assert ag.top_score([ctx(0.2), ctx(0.7), ctx(0.5)]) == 0.7


def test_top_score_prefers_rerank_over_rrf():
    """同一条里两个分数都在时，以精排为准（结果就是按精排排的）。"""
    c = {"content": "x", "source": "a.md", "type": "text",
         "rerank_score": 0.8, "rrf_score": 0.01}
    assert ag.top_score([c]) == 0.8


def test_top_score_falls_back_to_rrf():
    """精排关闭 / 降级时结果里没有 rerank_score，回退粗排分数。"""
    assert ag.top_score([ctx(0.016, key="rrf_score")]) == 0.016


def test_top_score_skips_entries_without_score():
    """没有分数字段（或为 None）的条目跳过，不崩。"""
    assert ag.top_score([{"content": "x", "source": "a.md"}]) == 0.0
    assert ag.top_score([{"content": "x", "rerank_score": None}]) == 0.0


def test_top_score_skips_non_dict_and_garbage():
    """脏数据（非 dict / 非数值）不得让整条链路崩。"""
    assert ag.top_score(["脏", None, 42, ctx(0.4)]) == 0.4
    assert ag.top_score([{"rerank_score": "不是数字"}, ctx(0.3)]) == 0.3


def test_top_score_accepts_numeric_string():
    assert ag.top_score([{"rerank_score": "0.75"}]) == 0.75


def test_top_score_all_zero():
    assert ag.top_score([ctx(0.0), ctx(0.0)]) == 0.0


# ---------- is_grounded ----------

def test_empty_contexts_is_not_grounded():
    assert ag.is_grounded([]) is False


def test_below_threshold_not_grounded():
    assert ag.is_grounded([ctx(0.12)], threshold=0.3) is False


def test_above_threshold_is_grounded():
    assert ag.is_grounded([ctx(0.62)], threshold=0.3) is True


def test_equal_threshold_is_grounded():
    """边界取 >=：恰好等于阈值算命中。"""
    assert ag.is_grounded([ctx(0.3)], threshold=0.3) is True


def test_gate_disabled_by_zero_threshold():
    """阈值 0 → 关闭门控，只要有检索结果就回答（与改动前行为一致）。"""
    assert ag.is_grounded([ctx(0.0001)], threshold=0.0) is True


def test_gate_disabled_by_negative_threshold():
    assert ag.is_grounded([ctx(0.0)], threshold=-1.0) is True


def test_disabled_gate_still_respects_empty():
    """关掉门控也不能把"什么都没检索到"当成能回答。"""
    assert ag.is_grounded([], threshold=0.0) is False


def test_uses_config_default(monkeypatch):
    monkeypatch.setattr(ag, "RETRIEVAL_MIN_SCORE", 0.9)
    assert ag.is_grounded([ctx(0.5)]) is False
    assert ag.is_grounded([ctx(0.95)]) is True


def test_explicit_threshold_overrides_config(monkeypatch):
    monkeypatch.setattr(ag, "RETRIEVAL_MIN_SCORE", 0.9)
    assert ag.is_grounded([ctx(0.5)], threshold=0.3) is True


def test_max_of_pool_decides():
    """只要候选池里有一条达标就算命中——门控看的是最高分，不是平均分。"""
    assert ag.is_grounded([ctx(0.9), ctx(0.01), ctx(0.01)], threshold=0.3) is True


# ---------- 真实分布夹具（kb 83cd2d0c，2026-09-14 实测精排 top1） ----------

ANSWERABLE = [0.620, 0.578]                     # 库内可答
NOT_ANSWERABLE = [0.125, 0.122, 0.091, 0.039,   # 闲聊 / 元问题 / 常识 / 库里没写
                  0.010, 0.001, 0.0005]         # 越界 / 天气 / 已知负样本


@pytest.mark.parametrize("score", ANSWERABLE)
def test_real_answerable_pass_default_threshold(score):
    assert ag.is_grounded([ctx(score)], threshold=0.3) is True


@pytest.mark.parametrize("score", NOT_ANSWERABLE)
def test_real_not_answerable_blocked_by_default_threshold(score):
    assert ag.is_grounded([ctx(score)], threshold=0.3) is False


def test_default_threshold_has_margin_on_both_sides():
    """默认 0.3 在实测分布两侧都留有余量（宁漏拒、不误拒）。"""
    margin_above = min(ANSWERABLE) - 0.3
    margin_below = 0.3 - max(NOT_ANSWERABLE)
    assert margin_above > 0.2   # 可答侧余量
    assert margin_below > 0.1   # 不可答侧余量
