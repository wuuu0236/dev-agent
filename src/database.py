"""
SQLite 数据库：知识库和文档的元数据管理

两张表：
  knowledge_bases — 知识库（id, name, description, created_at）
  documents       — 文档（id, kb_id, filename, file_size, chunk_count, status, created_at）

为什么用 SQLite：
  - 零配置，不需要安装数据库服务
  - 一个文件就是一个数据库，备份方便
  - 适合单机小规模应用（< 10万条）
  - 面试官一看就懂，不用解释 PostgreSQL/MySQL
"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
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
    conn.commit()
    conn.close()


# --- 知识库操作 ---

def create_kb(name: str, description: str = "") -> dict:
    """创建新知识库，返回它的数据"""
    conn = get_connection()
    kb_id = str(uuid.uuid4())[:8]  # 短 ID，方便显示
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO knowledge_bases (id, name, description, created_at) VALUES (?, ?, ?, ?)",
        (kb_id, name, description, now)
    )
    conn.commit()
    conn.close()
    return {"id": kb_id, "name": name, "description": description, "created_at": now}


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


def delete_kb(kb_id: str):
    """删除知识库及其所有文档。CASCADE 自动删除关联的 documents 记录。"""
    conn = get_connection()
    conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
    conn.commit()
    conn.close()


# --- 文档操作 ---

def add_document(kb_id: str, filename: str, file_size: int = 0) -> str:
    """添加文档记录，返回文档 ID"""
    conn = get_connection()
    doc_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO documents (id, kb_id, filename, file_size, status, created_at) VALUES (?, ?, ?, ?, 'processing', ?)",
        (doc_id, kb_id, filename, file_size, now)
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
