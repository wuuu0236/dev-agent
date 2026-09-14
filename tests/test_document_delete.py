"""文档删除的两侧一致性测试。

背景（这是个真实存在过的 bug）：
  documents 表（SQLite）存元数据，chunk 内容存 Chroma。原来的删除只跑了
  `DELETE FROM documents`，Chroma 里的 chunk 一个没动 —— 结果是文档从列表
  里消失、内容却仍然被检索到、答案里还在引用它（幽灵引用）。

这组测试锁住两条不变量：
  1. 按来源删除只影响该来源，不误伤同库其他文档
  2. 同来源重新入库是「替换」而非「叠加」——否则修订版变短时，尾部旧 chunk
     会因 id 没被覆盖而永久残留
"""
import pytest

from src import vector_store


class FakeCollection:
    """内存版 Chroma collection，只实现这条链路用到的三个方法。"""

    def __init__(self):
        self.docs: dict[str, str] = {}
        self.metas: dict[str, dict] = {}

    def add(self, ids, documents, embeddings, metadatas):
        for i, cid in enumerate(ids):
            self.docs[cid] = documents[i]
            self.metas[cid] = metadatas[i]

    def get(self, where=None, **kwargs):
        ids = [
            cid for cid, meta in self.metas.items()
            if not where or all(meta.get(k) == v for k, v in where.items())
        ]
        return {
            "ids": ids,
            "documents": [self.docs[i] for i in ids],
            "metadatas": [self.metas[i] for i in ids],
        }

    def delete(self, ids):
        for cid in ids:
            self.docs.pop(cid, None)
            self.metas.pop(cid, None)

    def count(self):
        return len(self.docs)


class FakeClient:
    def __init__(self):
        self.cols: dict[str, FakeCollection] = {}

    def get_collection(self, name):
        if name not in self.cols:
            raise ValueError(f"collection {name} does not exist")
        return self.cols[name]

    def create_collection(self, name, metadata=None):
        self.cols[name] = FakeCollection()
        return self.cols[name]

    def delete_collection(self, name):
        self.cols.pop(name, None)


@pytest.fixture
def fake_chroma(monkeypatch):
    """把 vector_store 的 chroma 客户端换成内存实现，并短路掉真实 embedding 调用。"""
    client = FakeClient()
    monkeypatch.setattr(vector_store, "_get_client", lambda: client)
    monkeypatch.setattr(vector_store, "_clear_query_cache", lambda kb_id: None)
    # 向量化是外部 API，测试里只需要「有向量」这个事实
    monkeypatch.setattr(
        vector_store, "embed_texts",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    return client


def _chunks(source: str, n: int) -> list[dict]:
    return [
        {"content": f"{source} 的第 {i} 段内容", "source": source,
         "page": 1, "chunk_index": i, "type": "text"}
        for i in range(n)
    ]


def _seed(client, kb_id: str, source: str, n: int):
    client.create_collection(f"kb_{kb_id}")
    vector_store.add_chunks(kb_id, _chunks(source, n))


# --- delete_chunks_by_source ---

def test_delete_removes_all_chunks_of_source(fake_chroma):
    _seed(fake_chroma, "kb1", "a.md", 3)
    assert fake_chroma.get_collection("kb_kb1").count() == 3

    removed = vector_store.delete_chunks_by_source("kb1", "a.md")

    assert removed == 3
    assert fake_chroma.get_collection("kb_kb1").count() == 0


def test_delete_only_touches_target_source(fake_chroma):
    """删 a.md 不能碰到 b.md —— 这是幽灵引用的反向风险：删多了等于数据丢失。"""
    _seed(fake_chroma, "kb1", "a.md", 3)
    vector_store.add_chunks("kb1", _chunks("b.md", 2))

    vector_store.delete_chunks_by_source("kb1", "a.md")

    remaining = fake_chroma.get_collection("kb_kb1").get()
    sources = {m["source"] for m in remaining["metadatas"]}
    assert sources == {"b.md"}
    assert len(remaining["ids"]) == 2


def test_delete_missing_collection_is_not_an_error(fake_chroma):
    """库不存在 = 没有 chunk 要删，返回 0 而不是抛异常（删文档流程不能被它打断）。"""
    assert vector_store.delete_chunks_by_source("never-existed", "a.md") == 0


def test_delete_unknown_source_returns_zero(fake_chroma):
    _seed(fake_chroma, "kb1", "a.md", 2)
    assert vector_store.delete_chunks_by_source("kb1", "ghost.md") == 0
    assert fake_chroma.get_collection("kb_kb1").count() == 2


# --- add_chunks 的 replace_source ---

def test_reupload_shorter_file_leaves_no_orphan_chunks(fake_chroma):
    """核心场景：10 块的文档重传成 6 块，第 7~10 块必须消失。

    chunk id 是 `{source}_chunk{i}`，若只做 upsert，新文件写不到 chunk6~9，
    这些旧内容会永远留在库里被检索到。
    """
    _seed(fake_chroma, "kb1", "a.md", 10)

    vector_store.add_chunks("kb1", _chunks("a.md", 6))

    col = fake_chroma.get_collection("kb_kb1")
    assert col.count() == 6
    assert sorted(col.docs) == [f"a.md_chunk{i}" for i in range(6)]


def test_reupload_does_not_affect_other_sources(fake_chroma):
    _seed(fake_chroma, "kb1", "a.md", 3)
    vector_store.add_chunks("kb1", _chunks("b.md", 2))

    vector_store.add_chunks("kb1", _chunks("a.md", 1))

    col = fake_chroma.get_collection("kb_kb1")
    assert col.count() == 3  # a 的 1 块 + b 的 2 块
    assert {m["source"] for m in col.metas.values()} == {"a.md", "b.md"}


def test_disabling_replace_demonstrates_the_orphan_bug(fake_chroma):
    """关掉替换后，变短的重传会残留旧 chunk —— 反证 replace_source 这一层的必要性。

    这不是"期望行为"，而是把 bug 本身钉在测试里：3 块变 1 块时，chunk1/chunk2
    因为 id 没被覆盖而留在库里，依然可被检索到。
    """
    _seed(fake_chroma, "kb1", "a.md", 3)

    vector_store.add_chunks("kb1", _chunks("a.md", 1), replace_source=False)

    col = fake_chroma.get_collection("kb_kb1")
    assert col.count() == 3                     # 1 个新的 + 2 个残留
    assert "a.md_chunk2" in col.docs            # 残留的旧内容


# --- database.delete_document ---

def test_delete_document_returns_record_and_removes_row(tmp_path, monkeypatch):
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "t.db")
    from src.database import (init_db, create_kb, add_document,
                              delete_document, list_documents)

    init_db()
    kb = create_kb("测试库")
    doc_id = add_document(kb["id"], "a.md", 1234)

    rec = delete_document(doc_id)

    assert rec is not None
    assert rec["filename"] == "a.md"       # 上层要靠它去清 Chroma
    assert rec["kb_id"] == kb["id"]
    assert list_documents(kb["id"]) == []


def test_delete_document_missing_id_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "t.db")
    from src.database import init_db, delete_document

    init_db()
    assert delete_document("no-such-id") is None
