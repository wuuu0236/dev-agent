"""
SQLite 数据库：知识库和文档的元数据管理

四张表：
  knowledge_bases — 知识库（id, name, description, owner_id, created_at）
  documents       — 文档（id, kb_id, filename, file_size, chunk_count, status, created_at）
  users           — 用户（id, username, password_hash, role, created_at）
  kb_permissions  — 知识库授权（kb_id, user_id, role），角色层级 owner > editor > viewer

users / kb_permissions 来自 docs/technical-optimization-plan.md 第一章。权限层刻意
保持最小：只有「归属 + 三种角色」，没有邀请链接、组织架构、SSO。

为什么用 SQLite：
  - 零配置，不需要安装数据库服务
  - 一个文件就是一个数据库，备份方便
  - 适合单机小规模应用（< 10万条）
  - 面试官一看就懂，不用解释 PostgreSQL/MySQL
"""
import hashlib
import os
import sqlite3
import uuid
from datetime import datetime

from src.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """获取数据库连接。
    每次调用都新建连接，因为 Streamlit 多线程环境下
    共享连接会出问题。
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # 让查询结果可以用名字访问，如 row['name']
    conn.execute("PRAGMA journal_mode=WAL")  # 写操作不阻塞读
    conn.execute("PRAGMA foreign_keys=ON")   # 启用外键约束
    return conn


def init_db():
    """初始化数据库表。首次运行时自动创建。
    在 app.py 启动时调用一次即可。

    幂等：建表全用 CREATE TABLE IF NOT EXISTS；列级变更走 _migrate_add_column。
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            kb_id       TEXT NOT NULL,
            filename    TEXT NOT NULL,
            file_size   INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'processing',
            created_at  TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            created_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_permissions (
            kb_id   TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role    TEXT NOT NULL DEFAULT 'viewer',
            PRIMARY KEY (kb_id, user_id),
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    _migrate_add_column(conn, "knowledge_bases", "owner_id", "TEXT")
    _migrate_add_column(conn, "documents", "content_hash", "TEXT DEFAULT ''")
    conn.commit()

    # users 为空 = 从单用户版本首次升级上来：建默认管理员并认领所有无主知识库。
    # 不认领的话，这些库的 owner_id 一直是 NULL，连管理员都"没有权限"看——
    # 老数据会集体变成孤儿。
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        _create_default_admin(conn)
        conn.commit()
    conn.close()


def _migrate_add_column(conn, table: str, column: str, decl: str):
    """幂等地给已有表加一列。

    SQLite 没有 `ADD COLUMN IF NOT EXISTS`，重复执行会抛 duplicate column name。
    这里先查 PRAGMA 再决定要不要 ALTER，而不是「catch 掉所有异常」——后者会把
    「表压根不存在」这类真错误一起吞掉，留下一半迁移成功的库。
    """
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _create_default_admin(conn):
    """建默认管理员，并把所有无主知识库划归给它。

    用户名 / 密码读 ADMIN_USERNAME / ADMIN_PASSWORD（默认 admin / changeme）。
    ⚠️ 默认密码是明文写死的弱口令，用途只有一个：本地首次升级时把老数据认领回来。
    部署到可能被外网访问的环境前必须改掉——它是整套权限体系里唯一的默认凭据。
    """
    from src.auth.service import hash_password  # 惰性：避开 database ←→ auth 循环 import

    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "changeme")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT OR IGNORE INTO users (id, username, password_hash, role, created_at) "
        "VALUES (?, ?, ?, 'admin', ?)",
        ("admin", username, hash_password(password), now),
    )
    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return  # 用户名被别的角色占了（UNIQUE 冲突）——不认领，避免把库挂到错误账号上
    admin_id = row["id"]
    conn.execute("UPDATE knowledge_bases SET owner_id = ? WHERE owner_id IS NULL", (admin_id,))
    for kb in conn.execute("SELECT id FROM knowledge_bases").fetchall():
        conn.execute(
            "INSERT OR REPLACE INTO kb_permissions (kb_id, user_id, role) VALUES (?, ?, 'owner')",
            (kb["id"], admin_id),
        )


# --- 知识库操作 ---

def create_kb(name: str, description: str = "", owner_id: str | None = None) -> dict:
    """创建新知识库，返回它的数据。

    owner_id = 创建者。**有登录用户时必须传**：`list_accessible_kbs` 的过滤条件是
    「owner_id 是我 或 kb_permissions 里有我」，不写 owner_id 就等于建了一个连
    创建者自己都看不见的库（只有 admin 例外）。默认 None 是给脚本与 Agent 路径
    （MCP 的 _ensure_kb）用的——那两条路径没有登录用户的概念。
    """
    conn = get_connection()
    kb_id = str(uuid.uuid4())[:8]  # 短 ID，方便显示
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO knowledge_bases (id, name, description, owner_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (kb_id, name, description, owner_id, now)
    )
    if owner_id:
        conn.execute(
            "INSERT OR REPLACE INTO kb_permissions (kb_id, user_id, role) VALUES (?, ?, 'owner')",
            (kb_id, owner_id),
        )
    conn.commit()
    conn.close()
    return {"id": kb_id, "name": name, "description": description,
            "owner_id": owner_id, "created_at": now}


def list_kbs() -> list[dict]:
    """列出所有知识库"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, description, created_at FROM knowledge_bases ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kb(kb_id: str) -> dict | None:
    """获取单个知识库"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _clear_answer_log(kb_id: str):
    """删库时连带清掉该库的问答日志。惰性 import + 吞异常，与 vector_store 的
    `_clear_query_cache` 同一套写法：日志是**旁路模块**，它导入失败 / 写不进去
    都不能拖垮"删库"这条主路径。"""
    try:
        from src.answer_log import clear_kb_log
        clear_kb_log(kb_id)
    except Exception:
        pass  # 日志模块不可用不影响删库


def delete_kb(kb_id: str):
    """删除知识库及其所有文档。CASCADE 自动删除关联的 documents 记录。

    连带清掉该库的问答日志——知识库没了，引用它的日志就是孤儿数据，留着只会
    让评估面板统计到一堆指向已删库的记录。清理放在这里（而不是调用方页面里）
    是为了保证**任何**调用方都清得掉，不会漏。
    """
    conn = get_connection()
    conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
    conn.commit()
    conn.close()
    _clear_answer_log(kb_id)


# --- 文档操作 ---

def add_document(kb_id: str, filename: str, file_size: int = 0,
                 content_hash: str = "") -> str:
    """添加文档记录，返回文档 ID。

    content_hash：整份文档内容的 SHA-256 指纹（compute_content_hash）。
    上传路径必须在调用本函数**之前**查重（check_duplicate）——
    指纹落了库再发现重复，就只能在"留垃圾行"和"删数据"之间二选一了。
    默认空串：脚本灌库路径（seed / build_eval_kb）不算指纹也能工作。
    """
    conn = get_connection()
    doc_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO documents (id, kb_id, filename, file_size, status, content_hash, created_at) "
        "VALUES (?, ?, ?, ?, 'processing', ?, ?)",
        (doc_id, kb_id, filename, file_size, content_hash, now)
    )
    conn.commit()
    conn.close()
    return doc_id


def update_document_status(doc_id: str, status: str, chunk_count: int = 0):
    """更新文档处理状态"""
    conn = get_connection()
    conn.execute(
        "UPDATE documents SET status = ?, chunk_count = ? WHERE id = ?",
        (status, chunk_count, doc_id)
    )
    conn.commit()
    conn.close()


def update_document_hash(doc_id: str, content_hash: str):
    """刷新文档的内容指纹。唯一调用方是 reindex：重建会按**当前**参数重新切分，
    指纹必须跟着刷新，否则重建后重传同一份原文不会被判重。"""
    conn = get_connection()
    try:
        conn.execute("UPDATE documents SET content_hash = ? WHERE id = ?", (content_hash, doc_id))
        conn.commit()
    finally:
        conn.close()


def compute_content_hash(chunks: list[dict]) -> str:
    """对分块结果计算整份文档的 SHA-256 内容指纹。

    只拼 chunk 内容、不含文件名 —— 去重判的是「内容一样」：
    同名不同内容要照常入库，不同名同内容才该被拦。
    收口在 database 而不是 vector_store：查重查的是 documents 表（SQLite），
    vector_store 是 Chroma 的封装，语义不该混。
    """
    combined = "\n".join(c["content"] for c in chunks)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def check_duplicate(kb_id: str, content_hash: str) -> bool:
    """同一知识库里是否已有相同内容指纹的 ready 文档（跨文件名）。

    只认 status='ready'：解析失败 / 空内容的记录不算"库里已有"。
    老数据该列默认 ''，不会被任何真实指纹命中，天然兼容。
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE kb_id = ? AND content_hash = ? AND status = 'ready'",
            (kb_id, content_hash),
        ).fetchone()
    finally:
        conn.close()
    return row[0] > 0


def get_document(doc_id: str) -> dict | None:
    """按 id 取单条文档记录（只读，不删）。

    存在的理由：删除接口的入参只有 doc_id，而权限是按**知识库**判的——
    必须先查这条记录属于哪个库，才能在做任何破坏性操作**之前**判定权限。
    用 delete_document 的返回值反推不行：那时数据已经没了。
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document(doc_id: str) -> dict | None:
    """删除文档记录，返回被删掉的记录（找不到则返回 None）。

    注意这只是 SQLite 侧的一半。chunk 内容存在 Chroma 里，调用方必须配合
    vector_store.delete_chunks_by_source(kb_id, filename) 一起清，
    否则会留下"列表里没了、检索还能搜到"的幽灵引用。
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return dict(row) if row else None


def list_documents(kb_id: str) -> list[dict]:
    """列出某个知识库的所有文档"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC", (kb_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kb_stats(kb_id: str) -> dict:
    """统计知识库信息：文档数、总 chunk 数"""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as doc_count, SUM(chunk_count) as total_chunks FROM documents WHERE kb_id = ? AND status = 'ready'",
        (kb_id,)
    ).fetchone()
    conn.close()
    return {"doc_count": row["doc_count"] or 0, "total_chunks": row["total_chunks"] or 0}
