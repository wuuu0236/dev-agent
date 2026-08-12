"""safety.py 三层安全审查测试（不碰数据库 / 不调 LLM，跨平台）。

为什么用 tmp_path 自建沙箱、monkeypatch 黑白名单，而不是写死 C:\\Windows：
  · CI 跑在 ubuntu，Windows 绝对路径不存在，abspath 后对不上黑名单
  · 更贴近"给 Agent 划沙箱"的设计意图：逻辑测试 = 在临时沙箱里自建危险区/安全区

沙箱布局：
  tmp/
    allow/    # 白名单区（模拟 C:\\Users）
      good.txt  .env  secret.txt  tokenization.py  password_manager.py
    danger/   # 黑名单区（模拟 C:\\Windows）
      evil.txt
    outside.txt           # 白名单外、黑名单外（已存在）
    evil_new.bat          # 白名单外（不存在）

每个用例对应一个真实攻击 / 误伤场景（见函数 docstring）。
"""
import os

import pytest

from src.tools import safety


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """搭一个"小沙箱"：allow 是白名单区，danger 是黑名单区。"""
    allow = tmp_path / "allow"
    danger = tmp_path / "danger"
    allow.mkdir()
    danger.mkdir()

    (allow / "good.txt").write_text("ok")
    (allow / ".env").write_text("KEY=secret")
    (allow / "secret.txt").write_text("top")
    (allow / "tokenization.py").write_text("def tokenize(): ...")
    (allow / "password_manager.py").write_text("...")
    (danger / "evil.txt").write_text("evil")
    (tmp_path / "outside.txt").write_text("outside")

    monkeypatch.setattr(safety, "ALLOWED_DIRS", [str(allow)])
    monkeypatch.setattr(safety, "DANGEROUS_PATHS", [str(danger)])
    return tmp_path, allow, danger


# ---------- 第一层：黑名单 ----------

def test_blacklist_rejects(sandbox):
    """黑名单：危险区里的文件一律拒绝，不管内容是什么。"""
    _, _, danger = sandbox
    ok, reason = safety.check_path_safety(str(danger / "evil.txt"))
    assert not ok
    assert "系统目录" in reason


def test_blacklist_subdir_rejects(sandbox):
    """黑名单：危险区的子目录同样拒绝。"""
    _, _, danger = sandbox
    ok, _ = safety.check_path_safety(str(danger / "nested" / "x.bat"))
    assert not ok


def test_path_within_boundary():
    """边界语义：'Windows' 只属于自身及其子目录，不误配 'WindowsApps'。
    修复前用 startswith，C:\\WindowsApps 这类普通应用目录会被误杀。"""
    base = "C:\\Windows"
    assert safety._path_within("C:\\Windows\\System32\\cmd.exe", base)   # 子路径
    assert safety._path_within("C:\\Windows", base)                      # 本身
    assert not safety._path_within("C:\\WindowsApps\\store", base)       # 前缀同名
    assert not safety._path_within("C:\\Windows.old\\x", base)           # 前缀同名


# ---------- 第二层：敏感文件 ----------

def test_sensitive_file_rejected(sandbox):
    """敏感文件：.env 和 secret.txt 都拦。"""
    _, allow, _ = sandbox
    for name in [".env", "secret.txt"]:
        ok, reason = safety.check_path_safety(str(allow / name))
        assert not ok, name
        assert "敏感文件" in reason, name


def test_sensitive_variant_rejected(sandbox):
    """.env.prod 这类变体名也拦（段名以"敏感词."开头）。"""
    _, allow, _ = sandbox
    (allow / ".env.prod").write_text("KEY=prod")
    ok, _ = safety.check_path_safety(str(allow / ".env.prod"))
    assert not ok


def test_sensitive_not_false_positive(sandbox):
    """敏感词边界：tokenization.py / password_manager.py 不该被 token/password 子串误伤。
    修复前用整串 in，正常代码文件会被拦，search_in_files 一搜就废。"""
    _, allow, _ = sandbox
    for name in ["tokenization.py", "password_manager.py"]:
        ok, _ = safety.check_path_safety(str(allow / name))
        assert ok, name


# ---------- 第三层：白名单 ----------

def test_whitelist_allows_inside(sandbox):
    """白名单：允许区内的普通文件放行。"""
    _, allow, _ = sandbox
    ok, _ = safety.check_path_safety(str(allow / "good.txt"))
    assert ok


def test_whitelist_rejects_outside(sandbox):
    """白名单：允许区外的文件（即使已存在）拒绝。"""
    tmp, _, _ = sandbox
    ok, reason = safety.check_path_safety(str(tmp / "outside.txt"))
    assert not ok
    assert "允许范围" in reason


def test_new_file_inside_allowed(sandbox):
    """新文件（不存在）在允许区内：放行。写文件场景不误杀。"""
    _, allow, _ = sandbox
    ok, _ = safety.check_path_safety(str(allow / "new.md"))
    assert ok


def test_new_file_outside_blocked(sandbox):
    """新文件（不存在）在允许区外：拦截。
    回归核心漏洞：旧代码白名单只在 os.path.exists 时检查，
    不存在的新文件会绕过白名单 → Agent 能在任意目录写文件。"""
    tmp, _, _ = sandbox
    ok, reason = safety.check_path_safety(str(tmp / "evil_new.bat"))
    assert not ok
    assert "允许范围" in reason


def test_path_traversal_normalized(sandbox):
    """路径穿越：allow/../../outside.txt 想借 .. 逃出沙箱，realpath 规范化后仍在白名单外。"""
    _, allow, _ = sandbox
    traversal = str(allow / ".." / ".." / "outside.txt")
    ok, _ = safety.check_path_safety(traversal)
    assert not ok


# ---------- 写检查 / 装饰器 ----------

def test_write_safety_blocks_config(sandbox):
    """写检查：禁止覆盖 settings.json 等配置文件。
    注意不用 .env 测：它在读检查的敏感文件层就被拦，到不了写检查这层。"""
    _, allow, _ = sandbox
    (allow / "settings.json").write_text("{}")
    ok, reason = safety.check_write_safety(str(allow / "settings.json"))
    assert not ok
    assert "配置文件" in reason


def test_write_safety_blocks_outside(sandbox):
    """写检查：白名单外路径同样拦。"""
    tmp, _, _ = sandbox
    ok, _ = safety.check_write_safety(str(tmp / "outside.txt"))
    assert not ok


def test_write_safety_allows_new_in_workdir(sandbox):
    """写检查：允许区内的新文件放行（不误杀正常写入）。"""
    _, allow, _ = sandbox
    ok, _ = safety.check_write_safety(str(allow / "new.txt"))
    assert ok


def test_decorator_intercepts(sandbox):
    """装饰器：路径不安全时不执行原函数，直接返回拦截信息。"""
    tmp, _, _ = sandbox

    @safety.require_safe_path("filepath")
    def fake_read(filepath: str) -> str:
        raise AssertionError("不该执行：不安全路径应在函数体内被拦截")

    out = fake_read(str(tmp / "outside.txt"))
    assert "安全拦截" in out


def test_decorator_passes_safe(sandbox):
    """装饰器：安全路径正常执行原函数。"""
    _, allow, _ = sandbox

    @safety.require_safe_path("filepath")
    def fake_read(filepath: str) -> str:
        return "content"

    assert fake_read(str(allow / "good.txt")) == "content"
