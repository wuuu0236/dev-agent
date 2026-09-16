"""知识库权限判断：谁能看、谁能改。

角色层级（数字越大权限越高）：
    viewer(0)  只读：看文档列表、提问、看问答日志
    editor(1)  可写：上传文档、重建索引、删文档
    owner(2)   可管理：分享给他人（= 知识库的创建者）

另有一个**全局**角色 admin：对所有知识库直接放行，用于默认管理员与运维。

判据一律「查库」而非「信 token」：token 里的 role 字段只用于前端展示，
真正判定时重新读 users 表——否则用户改了密码/被降权后，旧 token 在过期前
仍然带着旧权限。
"""
from src.database import get_connection

# 角色 → 权限等级。加新角色只改这一处。
ROLE_LEVEL = {"viewer": 0, "editor": 1, "owner": 2}

VALID_ROLES = tuple(ROLE_LEVEL)


def check_kb_access(user_id: str, kb_id: str, required: str = "viewer") -> bool:
    """user 对 kb 是否具备 required 级别的权限。"""
    if required not in ROLE_LEVEL:
        raise ValueError(f"未知的权限级别: {required}")

    conn = get_connection()
    try:
        user = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return False
        if user["role"] == "admin":
            return True  # 全局管理员：对任何库都有 owner 权限

        kb = conn.execute(
            "SELECT owner_id FROM knowledge_bases WHERE id = ?", (kb_id,)
        ).fetchone()
        if not kb:
            return False  # 库不存在 → 无权限（调用方再决定回 403 还是 404）
        if kb["owner_id"] == user_id:
            return True

        perm = conn.execute(
            "SELECT role FROM kb_permissions WHERE kb_id = ? AND user_id = ?",
            (kb_id, user_id),
        ).fetchone()
        if not perm:
            return False
        got = ROLE_LEVEL.get(perm["role"])
        return got is not None and got >= ROLE_LEVEL[required]
    finally:
        conn.close()


def list_accessible_kbs(user_id: str) -> list[dict]:
    """列出该用户能访问的所有知识库（自己建的 + 被分享的）。"""
    conn = get_connection()
    try:
        user = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return []
        if user["role"] == "admin":
            rows = conn.execute(
                "SELECT * FROM knowledge_bases ORDER BY created_at DESC"
            ).fetchall()
        else:
            # DISTINCT 必需：同一用户可能既有 owner_id 命中、又在 kb_permissions 里，
            # LEFT JOIN 会把这一行放大成两条。
            rows = conn.execute(
                """
                SELECT DISTINCT kb.* FROM knowledge_bases kb
                LEFT JOIN kb_permissions p ON kb.id = p.kb_id
                WHERE kb.owner_id = ? OR p.user_id = ?
                ORDER BY kb.created_at DESC
                """,
                (user_id, user_id),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def share_kb(kb_id: str, target_user_id: str, role: str):
    """把知识库分享给另一个用户（重复分享 = 改角色）。

    用 ValueError 而不是 assert：assert 在 `python -O` 下会被整个移除，
    那时候非法角色会一路写进数据库。参数校验不能依赖 assert。
    """
    if role not in ROLE_LEVEL:
        raise ValueError(f"非法角色: {role}，只允许 {', '.join(VALID_ROLES)}")
    conn = get_connection()
    try:
        target = conn.execute("SELECT id FROM users WHERE id = ?", (target_user_id,)).fetchone()
        if not target:
            raise ValueError(f"目标用户不存在: {target_user_id}")
        conn.execute(
            "INSERT OR REPLACE INTO kb_permissions (kb_id, user_id, role) VALUES (?, ?, ?)",
            (kb_id, target_user_id, role),
        )
        conn.commit()
    finally:
        conn.close()


def find_user_by_username(username: str) -> dict | None:
    """按用户名查（分享接口需要把用户名换成 user_id）。"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, role FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None
