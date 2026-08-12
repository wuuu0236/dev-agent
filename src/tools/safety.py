"""
工具安全审查 —— 防止 Agent 越权操作文件

多层检查链路：
  路径规范化 → 黑名单 → 敏感文件检测 → 白名单 → 放行/拒绝

面试时一句话：
  "在工具执行前加了一层安全检查，防止 Agent 误删系统文件或读取敏感信息。
   就像给 Agent 划了一个沙箱，只能在安全区域活动。"

三层各自防什么：
  ① 黑名单：绝对不能碰的系统目录（C:\\Windows、/etc…）
  ② 敏感文件：不能读的密钥/配置文件（.env、id_rsa…）
  ③ 白名单：只能在本机用户目录 + 项目目录里活动（真正的沙箱边界）
"""

import os

# ============================================================
# 规则配置
# ============================================================

# 绝对禁止的路径（系统目录）
DANGEROUS_PATHS = [
    "C:\\Windows",
    "C:\\Windows\\System32",
    "C:\\Windows\\SysWOW64",
    "/etc",
    "/sys",
    "/proc",
    "/boot",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
]

# 敏感文件名（禁止读取）
SENSITIVE_FILES = [
    ".env",
    ".gitconfig",
    "id_rsa",
    "id_ed25519",
    "known_hosts",
    "authorized_keys",
    "credentials",
    "secret",
    ".aws/credentials",
    ".ssh/",
    "password",
    "token",
]

# 允许的操作范围（白名单）
# 只允许在以下目录里操作，其他目录拒绝
ALLOWED_DIRS = [
    "C:\\Users",
    "/home",
    os.getcwd(),  # 当前项目目录
]

# ============================================================
# 匹配工具
# ============================================================

def _path_within(path: str, base: str) -> bool:
    """path 是否在 base 目录内（含 base 本身）。按目录段边界匹配。

    为什么不用 startswith：黑名单 "C:\\Windows" 用 startswith 会误杀真实存在的
    "C:\\WindowsApps"、"C:\\Windows.old"（两者是普通应用目录）。必须整段对齐：
      C:\\Windows\\System32\\cmd.exe ∈ C:\\Windows   → True
      C:\\WindowsApps\\x           ∈ C:\\Windows   → False
    """
    p = path.lower().rstrip("\\/")
    b = base.lower().rstrip("\\/")
    if p == b:
        return True
    return p.startswith(b + os.sep)


def _find_sensitive(path: str) -> str | None:
    """命中敏感名单返回命中的项，否则 None。按路径段匹配。

    为什么不用整串子串：敏感词 "token" 是子串，"tokenization.py" 也会被拦，
    正常代码文件被误伤。改成段匹配后：
      .env / id_rsa / secret.txt          → 命中（段名等于，或"敏感词.后缀"）
      tokenization.py / password_manager  → 放行（不是敏感文件名）
    带路径的敏感项（.aws/credentials）按段序列匹配。
    """
    parts = [p for p in path.replace("/", os.sep).lower().split(os.sep) if p]
    for sensitive in SENSITIVE_FILES:
        s = sensitive.lower().strip("\\/")
        if os.sep in s:
            # 子路径项：.aws/credentials → 检查这段序列是否出现在路径里
            seq = [p for p in s.split(os.sep) if p]
            if any(parts[i:i + len(seq)] == seq for i in range(len(parts) - len(seq) + 1)):
                return sensitive
            continue
        for part in parts:
            if part == s or part.startswith(s + "."):
                return sensitive
    return None

# ============================================================
# 安全检查函数
# ============================================================

def check_path_safety(path: str) -> tuple[bool, str]:
    """
    检查路径是否安全

    返回: (是否允许, 原因)
      - (True, "") → 安全，允许操作
      - (False, "原因") → 不安全，拒绝操作
    """
    # 规范化路径（处理相对路径和 .. 等）
    try:
        abs_path = os.path.abspath(os.path.realpath(path))
    except Exception:
        return False, f"无法解析路径: {path}"

    # 第一层：黑名单检查 —— 绝对不能碰
    for dangerous in DANGEROUS_PATHS:
        if _path_within(abs_path, dangerous):
            return False, f"系统目录禁止访问: {dangerous}"

    # 第二层：敏感文件检查 —— 不能读
    hit = _find_sensitive(abs_path)
    if hit is not None:
        return False, f"敏感文件禁止访问: 包含 '{hit}'"

    # 第三层：白名单检查 —— 只能在安全区域活动
    # 注意：无条件检查，不是"文件存在才查"。否则 Agent 往任意非系统目录写一个
    # 新文件（如 C:\ProgramData\x.bat，原本不存在）会绕过整个白名单。
    if not any(_path_within(abs_path, d) for d in ALLOWED_DIRS):
        return False, f"路径不在允许范围内，仅允许访问用户目录和项目目录"

    return True, ""


def check_write_safety(path: str) -> tuple[bool, str]:
    """
    写操作额外检查 —— 比读更严格
    防止：覆盖已有文件、写到系统目录
    """
    # 先用读检查
    safe, reason = check_path_safety(path)
    if not safe:
        return safe, reason

    abs_path = os.path.abspath(os.path.realpath(path))

    # 禁止覆盖项目配置文件
    config_files = ["settings.json", "docker-compose.yml", "Dockerfile", ".env"]
    if os.path.basename(abs_path) in config_files:
        return False, f"禁止覆盖配置文件: {os.path.basename(abs_path)}"

    return True, ""


# ============================================================
# 装饰器 —— 一行注解，自动安全检查
# ============================================================

def require_safe_path(path_arg: str = "filepath"):
    """
    装饰器：自动检查工具参数中的路径是否安全

    使用方式：
        @require_safe_path("filepath")
        def read_file(filepath: str) -> str:
            ...

    路径不安全时直接返回错误信息，不会执行原函数
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取要检查的路径
            path = kwargs.get(path_arg) or (args[0] if args else "")
            if path:
                safe, reason = check_path_safety(str(path))
                if not safe:
                    return f"安全拦截: {reason}"
            return func(*args, **kwargs)

        return wrapper

    return decorator
