"""reindex（重建索引）单元测试。

全部用 monkeypatch：不解析真文件、不切块、不调 embedding、不碰真库。
重点钉住三件事：
  · 缺原文的文档 → 记进 missing，且**绝不去动它的旧索引**（重建失败应等于"什么都没发生"）
  · 单个文档失败 → 不影响其他文档（逐文档替换，不做"整库清空再写"）
  · 成功路径的计数与状态回写正确
"""
from pathlib import Path

import pytest

import src.chunker as chunker
import src.config as cfg
import src.database as db
import src.parser as parser
import src.reindex as ri
import src.vector_store as vs

KB = "kb1"


def _doc(name):
    return {"id": f"d_{name}", "kb_id": KB, "filename": name, "file_size": 1,
            "chunk_count": 0, "status": "ready", "created_at": "2026-09-14"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """假环境：临时 uploads 目录 + 假的 db / parser / chunker / vector_store。"""
    monkeypatch.setattr(cfg, "UPLOAD_DIR", tmp_path / "uploads")

    box = {
        "docs": [],
        "added": [],          # add_chunks 收到 (kb_id, chunks)
        "status": [],         # update_document_status 收到 (doc_id, status, count)
        "parsed": [],         # parse_file 收到路径
        "empty_parse": set(),  # 这些文件的解析结果为空
        "empty_chunk": set(),  # 这些文件切不出块
        "raise_on_add": set(),  # 这些文件在入库时抛异常
    }

    def fake_parse(path):
        name = Path(path).name
        box["parsed"].append(name)
        if name in box["empty_parse"]:
            return []
        return [{"text": f"内容-{name}", "page": None, "source": name}]

    def fake_chunk(parsed):
        if not parsed:
            return []
        name = parsed[0]["source"]
        if name in box["empty_chunk"]:
            return []
        return [{"content": parsed[0]["text"], "source": name,
                 "page": None, "chunk_index": 0, "type": "text"}]

    def fake_add(kb_id, chunks, replace_source=True):
        name = chunks[0]["source"]
        if name in box["raise_on_add"]:
            raise RuntimeError("embedding 服务挂了")
        box["added"].append((kb_id, chunks))

    monkeypatch.setattr(db, "list_documents", lambda kb_id: list(box["docs"]))
    monkeypatch.setattr(
        db, "update_document_status",
        lambda doc_id, status, count=0: box["status"].append((doc_id, status, count)),
    )
    monkeypatch.setattr(parser, "parse_file", fake_parse)
    monkeypatch.setattr(chunker, "chunk_parsed", fake_chunk)
    monkeypatch.setattr(vs, "add_chunks", fake_add)
    return box


def _put(env, filename):
    """把原始文件放进该知识库的上传目录。"""
    (cfg.kb_upload_dir(KB) / filename).write_text("x", encoding="utf-8")


# ---------- 空库 ----------

def test_no_documents(env):
    assert ri.rebuild_kb(KB) == {"total": 0, "ok": 0, "chunks": 0, "missing": [], "failed": []}


# ---------- 缺原文（改造前入库的老数据） ----------

def test_missing_original_is_reported_and_untouched(env):
    """原文不在 → 记进 missing，且绝不动它的旧索引。"""
    env["docs"] = [_doc("gone.md")]
    r = ri.rebuild_kb(KB)
    assert r["missing"] == ["gone.md"]
    assert r["ok"] == 0
    assert env["added"] == []           # 关键：没有写库，旧 chunk 原样保留
    assert env["parsed"] == []          # 也不会去解析一个不存在的文件


def test_missing_mixed_with_ok(env):
    """一缺一有：只重建有原文的那个。"""
    env["docs"] = [_doc("a.md"), _doc("b.md")]
    _put(env, "a.md")
    r = ri.rebuild_kb(KB)
    assert (r["ok"], r["total"], r["missing"]) == (1, 2, ["b.md"])
    assert [c[0]["source"] for _, c in env["added"]] == ["a.md"]


def test_rebuildable_docs_splits_ready_and_missing(env):
    env["docs"] = [_doc("a.md"), _doc("b.md"), _doc("c.md")]
    _put(env, "a.md")
    _put(env, "c.md")
    ready, missing = ri.rebuildable_docs(KB)
    assert ready == ["a.md", "c.md"]
    assert missing == ["b.md"]


# ---------- 正常重建 ----------

def test_successful_rebuild(env):
    env["docs"] = [_doc("a.md")]
    _put(env, "a.md")
    r = ri.rebuild_kb(KB)
    assert (r["ok"], r["total"], r["chunks"]) == (1, 1, 1)
    assert r["missing"] == [] and r["failed"] == []
    assert env["added"][0][0] == KB
    assert env["status"] == [("d_a.md", "ready", 1)]


def test_progress_callback_receives_each_file(env):
    env["docs"] = [_doc("a.md"), _doc("b.md")]
    _put(env, "a.md")
    _put(env, "b.md")
    seen = []
    ri.rebuild_kb(KB, progress=seen.append)
    assert seen == ["[1/2] a.md", "[2/2] b.md"]


def test_progress_optional(env):
    """不传 progress 也能正常跑。"""
    env["docs"] = [_doc("a.md")]
    _put(env, "a.md")
    assert ri.rebuild_kb(KB)["ok"] == 1


# ---------- 失败路径 ----------

def test_empty_parse_marks_empty_and_skips_write(env):
    env["docs"] = [_doc("a.md")]
    _put(env, "a.md")
    env["empty_parse"].add("a.md")
    r = ri.rebuild_kb(KB)
    assert r["failed"] == [("a.md", "没有可用的文本内容")]
    assert env["status"] == [("d_a.md", "empty", 0)]
    assert env["added"] == []


def test_empty_chunks_marks_empty(env):
    env["docs"] = [_doc("a.md")]
    _put(env, "a.md")
    env["empty_chunk"].add("a.md")
    r = ri.rebuild_kb(KB)
    assert [f[0] for f in r["failed"]] == ["a.md"]
    assert env["status"] == [("d_a.md", "empty", 0)]


def test_one_failure_does_not_stop_others(env):
    """嵌入失败只影响那一个文档，其他照常重建（不做整库清空）。"""
    env["docs"] = [_doc("a.md"), _doc("b.md"), _doc("c.md")]
    for n in ("a.md", "b.md", "c.md"):
        _put(env, n)
    env["raise_on_add"].add("b.md")

    r = ri.rebuild_kb(KB)
    assert r["ok"] == 2
    assert [f[0] for f in r["failed"]] == ["b.md"]
    assert "embedding 服务挂了" in r["failed"][0][1]
    assert [c[0]["source"] for _, c in env["added"]] == ["a.md", "c.md"]
    # 失败的那个不写状态（保持原样，等重试）
    assert ("d_b.md", "ready", 1) not in env["status"]


def test_failed_document_keeps_old_index(env):
    """失败的文档不留任何写入痕迹——旧索引不被动过。"""
    env["docs"] = [_doc("a.md")]
    _put(env, "a.md")
    env["raise_on_add"].add("a.md")
    ri.rebuild_kb(KB)
    assert env["added"] == []
    assert env["status"] == []
