"""多用户与权限（docs/technical-optimization-plan.md 第一章）行为测试。

覆盖三块：
  ① 认证：注册 / 登录 / token 签发与校验，失败路径不泄露用户名存在性
  ② 授权：owner > editor > viewer 三层 + 全局 admin 放行 + 列表隔离
  ③ 迁移：从单用户版本首次升级时，默认管理员认领所有无主知识库

隔离方式：monkeypatch `src.database.DB_PATH` 指向临时文件（get_connection 每次新建
连接、init_db 幂等，天然支持这种替换）；不调 embedding / LLM。
"""
import pytest

import src.database as db
from src.auth import permissions as perm
from src.auth import service as svc
from src.auth.deps import get_current_user
from fastapi import HTTPException

ADMIN_PASSWORD = "test-admin-pass"


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    """指向临时库并完成初始化（建表 + 默认 admin 迁移）。"""
    db_file = tmp_path / "test_auth.db"
    monkeypatch.setattr(db, "DB_PATH", db_file)
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN_PASSWORD)
    # bcrypt 哈希只受 rounds 影响耗时、不影响算法行为；默认 12 轮在本测试集里
    # 要算 ~10 次哈希（建 admin + 每个注册用户），降到 4 轮把秒级压到毫秒级。
    # ⚠️ 必须先存原函数：svc.bcrypt 就是 bcrypt 模块本身，替换后的 lambda 若
    # 再调 bcrypt.gensalt 会调到自己 → RecursionError。
    import bcrypt

    _real_gensalt = bcrypt.gensalt
    monkeypatch.setattr(
        svc.bcrypt, "gensalt", lambda rounds=12: _real_gensalt(rounds=4)
    )
    db.init_db()
    return db_file


def _sql(auth_env, stmt, args=()):
    conn = db.get_connection()
    out = conn.execute(stmt, args).fetchall()
    conn.commit()
    conn.close()
    return out


def _register(username):
    return svc.register(username, "pass123")


# ---------- ① 认证 ----------

class TestRegister:
    def test_register_success(self, auth_env):
        info = _register("alice")
        assert info["username"] == "alice" and info["role"] == "user"
        rows = _sql(auth_env, "SELECT password_hash FROM users WHERE username='alice'")
        stored = rows[0]["password_hash"]
        assert stored != "pass123" and stored.startswith("$2")  # 存的是 bcrypt 哈希

    def test_register_duplicate_raises(self, auth_env):
        _register("bob")
        with pytest.raises(ValueError, match="用户名已存在"):
            _register("bob")


class TestLogin:
    def test_login_returns_token_and_decode_roundtrip(self, auth_env):
        created = _register("carol")
        result = svc.login("carol", "pass123")
        assert result is not None
        payload = svc.decode_token(result["token"])
        assert payload["sub"] == created["id"]
        assert payload["username"] == "carol"
        assert result["user"]["role"] == "user"

    def test_login_wrong_password_returns_none(self, auth_env):
        _register("dave")
        assert svc.login("dave", "wrong-pass") is None

    def test_login_unknown_user_returns_none(self, auth_env):
        # 与密码错误返回同样的 None：不区分「用户不存在」和「密码错」，
        # 否则登录接口就成了用户名枚举器。
        assert svc.login("nobody-here", "whatever") is None


class TestToken:
    def test_decode_rejects_tampered_token(self, auth_env):
        token = svc.create_token("u1", "u1", "user")
        assert svc.decode_token(token + "x") is None
        assert svc.decode_token("not-a-jwt") is None

    def test_verify_password_survives_corrupt_hash(self, auth_env):
        # 库里一条脏数据不该让登录接口 500
        assert svc.verify_password("x", "not-a-bcrypt-hash") is False

    def test_get_user_hides_password_hash(self, auth_env):
        created = _register("erin")
        info = svc.get_user(created["id"])
        assert info["username"] == "erin" and "password_hash" not in info


class TestAuthHeader:
    def test_missing_or_malformed_header_is_401(self, auth_env):
        with pytest.raises(HTTPException) as ei:
            get_current_user(None)
        assert ei.value.status_code == 401
        with pytest.raises(HTTPException) as ei:
            get_current_user("Token abc")  # 不是 Bearer
        assert ei.value.status_code == 401
        with pytest.raises(HTTPException) as ei:
            get_current_user("Bearer garbage")
        assert ei.value.status_code == 401


# ---------- ② 授权 ----------

class TestKbAccess:
    def test_owner_has_full_access(self, auth_env):
        uid = _register("owner1")["id"]
        kb = db.create_kb("kb-a", owner_id=uid)
        assert perm.check_kb_access(uid, kb["id"], "owner") is True

    def test_viewer_can_read_but_not_edit(self, auth_env):
        uid = _register("owner2")["id"]
        vid = _register("viewer2")["id"]
        kb = db.create_kb("kb-b", owner_id=uid)
        perm.share_kb(kb["id"], vid, "viewer")
        assert perm.check_kb_access(vid, kb["id"], "viewer") is True
        assert perm.check_kb_access(vid, kb["id"], "editor") is False

    def test_unrelated_user_denied(self, auth_env):
        uid = _register("owner3")["id"]
        other = _register("stranger")["id"]
        kb = db.create_kb("kb-c", owner_id=uid)
        assert perm.check_kb_access(other, kb["id"], "viewer") is False

    def test_admin_bypasses_every_kb(self, auth_env):
        uid = _register("owner4")["id"]
        kb = db.create_kb("kb-d", owner_id=uid)
        assert perm.check_kb_access("admin", kb["id"], "owner") is True

    def test_unknown_required_level_raises(self, auth_env):
        with pytest.raises(ValueError, match="未知的权限级别"):
            perm.check_kb_access("admin", "any-kb", "root")


class TestAccessibleList:
    def test_list_only_own_and_shared(self, auth_env):
        u1 = _register("list-a")["id"]
        u2 = _register("list-b")["id"]
        u3 = _register("list-c")["id"]
        kb1 = db.create_kb("mine", owner_id=u1)["id"]
        perm.share_kb(kb1, u2, "viewer")

        names = lambda uid: {k["name"] for k in perm.list_accessible_kbs(uid)}
        assert names(u1) == {"mine"}   # 自己建的
        assert names(u2) == {"mine"}   # 被分享的
        assert names(u3) == set()      # 无关者看不到

    def test_share_kb_validation_and_role_change(self, auth_env):
        uid = _register("shr-a")["id"]
        vid = _register("shr-b")["id"]
        kb = db.create_kb("shr", owner_id=uid)["id"]
        with pytest.raises(ValueError, match="非法角色"):
            perm.share_kb(kb, vid, "root")
        with pytest.raises(ValueError, match="目标用户不存在"):
            perm.share_kb(kb, "ghost", "viewer")
        perm.share_kb(kb, vid, "viewer")
        perm.share_kb(kb, vid, "owner")  # 重复分享 = 改角色
        assert perm.check_kb_access(vid, kb, "editor") is True


# ---------- ③ 迁移 ----------

class TestMigration:
    def test_fresh_init_creates_default_admin(self, auth_env):
        rows = _sql(auth_env, "SELECT id, role FROM users WHERE role='admin'")
        assert len(rows) == 1 and rows[0]["id"] == "admin"

    def test_orphan_kbs_get_claimed_on_upgrade(self, auth_env):
        """模拟单用户老库升级：库里已有知识库、但 users 为空（owner 全是 NULL）。"""
        kb = db.create_kb("legacy", owner_id=None)  # 老数据：无主
        _sql(auth_env, "DELETE FROM users")         # 抹掉用户 → 下次 init_db 视为首次升级
        _sql(auth_env, "DELETE FROM kb_permissions")
        db.init_db()                                # 再走一次初始化
        row = _sql(auth_env, "SELECT owner_id FROM knowledge_bases WHERE id=?", (kb["id"],))
        assert row[0]["owner_id"] == "admin"
        perm_row = _sql(
            auth_env,
            "SELECT role FROM kb_permissions WHERE kb_id=? AND user_id='admin'", (kb["id"],),
        )
        assert perm_row[0]["role"] == "owner"
