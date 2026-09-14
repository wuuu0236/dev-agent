"""Embedding 分批与重试测试。

背景：原来是把整份文件的 chunk 一次性发给 API。一本 500 页 PDF 约 2000 个
chunk，单请求体积 / 超时 / 限流三样随便撞一个就整体失败。改成「分批 + 按批
重试」之后，要锁住三条不变量：

  1. 批次划分正确（每批不超过上限，最后一批收尾）
  2. 返回顺序与输入顺序严格一致（错位会让 chunk 和向量张冠李戴）
  3. 重试用尽才抛异常，不静默返回半截结果
"""
import pytest

from src import embeddings


@pytest.fixture
def no_sleep(monkeypatch):
    """重试的退避等待在测试里跳过，否则要真等 3 秒。"""
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)


def test_empty_input_returns_empty(monkeypatch):
    monkeypatch.setattr(embeddings, "_embed_one_batch",
                        lambda t: pytest.fail("空输入不应发起调用"))
    assert embeddings.embed_texts([]) == []


def test_small_input_uses_single_call(monkeypatch, no_sleep):
    calls = []

    def fake(batch):
        calls.append(len(batch))
        return [[0.0] for _ in batch]

    monkeypatch.setattr(embeddings, "EMBED_BATCH_SIZE", 64)
    monkeypatch.setattr(embeddings, "_embed_one_batch", fake)

    out = embeddings.embed_texts([f"t{i}" for i in range(10)])

    assert calls == [10]
    assert len(out) == 10


def test_large_input_is_split_into_batches(monkeypatch, no_sleep):
    calls = []

    def fake(batch):
        calls.append(len(batch))
        return [[0.0] for _ in batch]

    monkeypatch.setattr(embeddings, "EMBED_BATCH_SIZE", 64)
    monkeypatch.setattr(embeddings, "_embed_one_batch", fake)

    out = embeddings.embed_texts([f"t{i}" for i in range(150)])

    assert calls == [64, 64, 22]   # 最后一批收尾，不丢不重
    assert len(out) == 150


def test_result_order_matches_input_order(monkeypatch, no_sleep):
    """跨批拼接后顺序必须是输入顺序，否则向量跟 chunk 会错配。"""
    def fake(batch):
        return [[float(t[1:])] for t in batch]

    monkeypatch.setattr(embeddings, "EMBED_BATCH_SIZE", 3)
    monkeypatch.setattr(embeddings, "_embed_one_batch", fake)

    out = embeddings.embed_texts([f"t{i}" for i in range(8)])

    assert [v[0] for v in out] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_retry_then_success(monkeypatch, no_sleep):
    """第一次失败、第二次成功 —— 应重试而不是直接抛。"""
    state = {"n": 0}

    def flaky(batch):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("429 rate limited")
        return [[0.0] for _ in batch]

    monkeypatch.setattr(embeddings, "EMBED_MAX_RETRIES", 3)
    monkeypatch.setattr(embeddings, "_embed_one_batch", flaky)

    out = embeddings.embed_texts(["只有一个"])

    assert len(out) == 1
    assert state["n"] == 2


def test_retry_exhausted_raises(monkeypatch, no_sleep):
    """重试用尽必须抛异常 —— 静默返回半截会让文档被误标为 ready。"""
    state = {"n": 0}

    def always_fail(batch):
        state["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(embeddings, "EMBED_MAX_RETRIES", 3)
    monkeypatch.setattr(embeddings, "_embed_one_batch", always_fail)

    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["a"])
    assert state["n"] == 3   # 试满 3 次


def test_retry_count_is_configurable(monkeypatch, no_sleep):
    state = {"n": 0}

    def always_fail(batch):
        state["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(embeddings, "EMBED_MAX_RETRIES", 1)
    monkeypatch.setattr(embeddings, "_embed_one_batch", always_fail)

    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["a"])
    assert state["n"] == 1   # 只试一次，不重试


def test_batch_failure_does_not_return_partial_result(monkeypatch, no_sleep):
    """第二批彻底失败 → 整体抛异常，不能返回"第一批的 3 条"当成功。

    注意是按**批次内容**判断失败，不是按调用次数——否则重试时计数变化会让
    第二批"第二次就成功"，测不出真正要测的东西。
    """
    def fake(batch):
        if batch and batch[0] == "t3":     # 第二批（t3/t4/t5）永远失败
            raise RuntimeError("second batch down")
        return [[0.0] for _ in batch]

    monkeypatch.setattr(embeddings, "EMBED_BATCH_SIZE", 3)
    monkeypatch.setattr(embeddings, "EMBED_MAX_RETRIES", 2)
    monkeypatch.setattr(embeddings, "_embed_one_batch", fake)

    with pytest.raises(RuntimeError):
        embeddings.embed_texts([f"t{i}" for i in range(7)])
