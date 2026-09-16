"""内容去重（docs/technical-optimization-plan.md 第八章）行为测试。

核心约束：
  · 指纹判「内容」不判「文件名」——同内容不同名要拦，同名不同内容要放行
  · 只认 status='ready' 的记录——解析失败的半成品不算"库里已有"
  · API 层重复上传返回 409，且**不落 documents 记录**（挡在 add_document 之前）
"""
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.config as cfg
import src.database as db
import src.vector_store as vs


@pytest.fixture
def dedup_env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_dedup.db")
    # 上传端点会把原文写进 uploads 目录 —— 指到临时目录，别污染真实 data/
    monkeypatch.setattr(cfg, "UPLOAD_DIR", tmp_path / "uploads")
    db.init_db()
    return tmp_path


def _chunks(*texts):
    return [{"content": t, "source": "x", "page": 1, "chunk_index": i} for i, t in enumerate(texts)]


class TestFingerprint:
    def test_same_content_different_names_share_hash(self, dedup_env):
        """指纹只拼 chunk 内容：同内容换文件名，指纹必须一致（否则查重形同虚设）。"""
        a = db.compute_content_hash(_chunks("段落一", "段落二"))
        b = db.compute_content_hash([{"content": "段落一", "source": "别的文件"},
                                     {"content": "段落二", "source": "别的文件"}])
        assert a == b

    def test_different_content_different_hash(self, dedup_env):
        assert db.compute_content_hash(_chunks("内容甲")) != db.compute_content_hash(_chunks("内容乙"))


class TestCheckDuplicate:
    def test_duplicate_detected_after_ready(self, dedup_env):
        kb = db.create_kb("kb-a")["id"]
        h = db.compute_content_hash(_chunks("重复内容"))
        assert db.check_duplicate(kb, h) is False
        doc = db.add_document(kb, "a.txt", content_hash=h)
        # 关键：processing 状态还不算 —— 只有成功入库（ready）后才拦
        assert db.check_duplicate(kb, h) is False
        db.update_document_status(doc, "ready", 2)
        assert db.check_duplicate(kb, h) is True

    def test_no_cross_kb_leak(self, dedup_env):
        h = db.compute_content_hash(_chunks("内容"))
        db.create_kb("kb-1")["id"]
        kb2 = db.create_kb("kb-2")["id"]
        doc = db.add_document(kb2, "b.txt", content_hash=h)
        db.update_document_status(doc, "ready")
        # 同内容在别的库不算重复：去重范围是**知识库内**
        assert db.check_duplicate(kb2, h) is True
        kb1 = [k["id"] for k in db.list_kbs() if k["name"] == "kb-1"][0]
        assert db.check_duplicate(kb1, h) is False

    def test_add_document_persists_hash(self, dedup_env):
        kb = db.create_kb("kb-c")["id"]
        db.add_document(kb, "c.txt", content_hash="deadbeef")
        # 走 SQL 直查（database 没有按文件名取文档的函数，别为测试加生产代码）
        conn = sqlite3.connect(str(db.DB_PATH))
        conn.row_factory = sqlite3.Row
        stored = conn.execute("SELECT content_hash FROM documents WHERE filename='c.txt'").fetchone()
        conn.close()
        assert stored["content_hash"] == "deadbeef"


class TestUploadEndpoint:
    @pytest.fixture
    def client(self, dedup_env, monkeypatch):
        # add_chunks 需要 embedding API —— 去重测试只关心 SQLite 侧，
        # 用假实现替换（web_api.upload 是函数内 import，每次从模块取属性，patch 生效）
        monkeypatch.setattr(vs, "add_chunks", lambda kb_id, chunks: None)
        from src.api.web_api import router

        app = FastAPI()
        app.include_router(router)
        c = TestClient(app)
        # 模块一之后全部端点要登录；upload 还要 editor 权限 —— 注册用户自己建库当 owner
        import uuid as _uuid

        uname = "u" + _uuid.uuid4().hex[:8]
        uid = c.post("/api/auth/register",
                     json={"username": uname, "password": "secret123"}).json()["id"]
        token = c.post("/api/auth/login",
                       json={"username": uname, "password": "secret123"}).json()["token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        return c, uid

    def _upload(self, client, kb_id, name, body):
        return client.post(f"/api/kbs/{kb_id}/upload", files={"file": (name, body, "text/plain")})

    def test_duplicate_upload_returns_409_and_skips_record(self, client, dedup_env):
        c, uid = client
        kb = db.create_kb("kb-dup", owner_id=uid)["id"]
        body = "同一份内容的文本文件，用来触发内容级查重逻辑。".encode("utf-8")
        r1 = self._upload(c, kb, "first.txt", body)
        assert r1.status_code == 200, r1.text
        r2 = self._upload(c, kb, "second.txt", body)  # 不同文件名、相同内容
        assert r2.status_code == 409
        conn = sqlite3.connect(str(db.DB_PATH))
        n = conn.execute("SELECT COUNT(*) FROM documents WHERE status='ready'").fetchone()[0]
        conn.close()
        assert n == 1, "重复上传不得落第二条 ready 记录"

    def test_same_name_upload_is_not_blocked_by_dedup(self, client, dedup_env):
        """同名重传（内容不同）必须照常入库——这是"覆盖更新"的老行为，不能被查重误伤。"""
        c, uid = client
        kb = db.create_kb("kb-same", owner_id=uid)["id"]
        r1 = self._upload(c, kb, "doc.txt", "第一版内容".encode("utf-8"))
        r2 = self._upload(c, kb, "doc.txt", "第二版内容，和第一版不同".encode("utf-8"))
        assert r1.status_code == 200 and r2.status_code == 200
