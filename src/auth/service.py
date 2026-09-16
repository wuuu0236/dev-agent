"""认证服务：密码哈希 / JWT 签发与校验 / 注册 / 登录。

设计约束：
  · 不引第三方认证框架——只做「够用的最小实现」，别让登录本身变成新的复杂度来源。
  · 凭据相关配置全部集中在 src/config.py（本模块不含魔数）。
  · 惰性 import database 的方向是反的（database 首次升级时要调 hash_password），
    所以 database 那边用函数内 import，本模块顶层 import 它是安全的。

密码存储：bcrypt，自带随机盐，哈希串里已含 salt，因此 users 表不需要单独的 salt 列。
"""
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from src.config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, JWT_SECRET
from src.database import get_connection

# bcrypt 只处理前 72 字节，超出部分被**静默丢弃**——不显式截断的话，
# 两个前 72 字节相同的长密码会被判为同一个密码。这里主动截断并留下痕迹，
# 至少行为是确定的（而不是"看 bcrypt 版本"）。
_BCRYPT_MAX_BYTES = 72


def _to_bytes(plain: str) -> bytes:
    return plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_to_bytes(plain), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """校验密码。哈希串损坏时返回 False，不把异常抛给调用方——
    登录接口不该因为库里有一条脏数据就 500。"""
    try:
        return bcrypt.checkpw(_to_bytes(plain), hashed.encode())
    except (ValueError, TypeError):
        return False


def create_token(user_id: str, username: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """解 token。过期 / 签名不对 / 格式错统一返回 None。

    只 catch PyJWTError 而不是 Exception：PyJWT 的所有失败都继承自它，
    而其它异常（比如我们自己的 bug）应该照常暴露出来。
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def register(username: str, password: str) -> dict:
    """注册新用户。用户名已存在时抛 ValueError（由入口层转 409/400）。"""
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f"用户名已存在: {username}")
        user_id = str(uuid.uuid4())[:8]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, created_at) "
            "VALUES (?, ?, ?, 'user', ?)",
            (user_id, username, hash_password(password), now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": user_id, "username": username, "role": "user"}


def login(username: str, password: str) -> dict | None:
    """校验用户名密码。成功返回 {token, user}，失败返回 None。

    失败原因不区分「用户不存在」和「密码错」——对外都是 None，
    调用方也只回一句"用户名或密码错误"，避免泄露哪些用户名存在。
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return {
        "token": create_token(row["id"], row["username"], row["role"]),
        "user": {"id": row["id"], "username": row["username"], "role": row["role"]},
    }


def get_user(user_id: str) -> dict | None:
    """按 id 取用户（不含 password_hash）。"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, role, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None
