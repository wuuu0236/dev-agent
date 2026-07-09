"""
工具安全审查 —— 防止 Agent 越权操作文件

多层检查链路：
  规则过滤 → 路径白名单 → 敏感文件检测 → 放行/拒绝

面试时一句话：
  "在工具执行前加了一层安全检查，防止 Agent 误删系统文件或读取敏感信息。
   就像给 Agent 划了一个沙箱，只能在安全区域活动。"
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
        if abs_path.lower().startswith(dangerous.lower()):
            return False, f"系统目录禁止访问: {dangerous}"

    # 第二层：敏感文件检查 —— 不能读
    filename = os.path.basename(abs_path).lower()
    for sensitive in SENSITIVE_FILES:
        if sensitive.lower() in abs_path.lower():
            return False, f"敏感文件禁止访问: 包含 '{sensitive}'"

    # 第三层：白名单检查 —— 允许在安全区域活动
    # 如果路径已存在，检查是否在白名单内
    if os.path.exists(abs_path):
        in_allowed = any(
            abs_path.lower().startswith(d.lower()) for d in ALLOWED_DIRS
        )
        if not in_allowed:
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
