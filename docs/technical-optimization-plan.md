# DataLens RAG 技术优化方案

> **读者**：AI coding agent（实现者）。
> **前提**：你已经能读通现有代码。本文档告诉你**改什么、怎么改、改到什么程度算完成**。
> **原则**：每个模块独立开关、独立测试、独立可回退。任何模块失败不阻断主链路。

---

## 全局约定

### 你不许做的事

1. 不修改 `src/hybrid_retriever.py` 的 `HybridRetriever.search()` 对外签名。
2. 不修改 `src/rag_qa.py` 的 `rag_query()` 和 `stream_rag_query()` 的返回值结构。
3. 不删除任何现有的 `try/except` 降级逻辑。
4. 不引入新的 Python 框架（不换 FastAPI、不换 SQLite、不换 Chroma）。
5. 不修改 `tests/` 目录下已有的测试用例（只允许新增）。
6. 每个新模块都必须能在没有 API key 的环境下被 import 而不报错。
7. 不在模块顶层直接创建 OpenAI 客户端，必须用惰性单例。
8. 不在 `src/` 目录下新增文件时使用绝对导入以外的路径写法，统一 `from src.xxx import ...`。

### 新增配置统一放 `src/config.py`

所有新配置变量加在文件末尾。格式：

```python
FEATURE_NAME = os.getenv("FEATURE_NAME", "默认值")
```

同步更新 `.env.example`（每个新 key 加一行带中文注释）和 `tests/test_env_example.py`。

---

## 一、多用户与权限模块

### 1.1 现状

当前系统是单用户的：任何请求都可以操作任何知识库，没有登录，没有权限检查。`database.py` 里 `knowledge_bases` 表没有 `owner_id` 字段。

### 1.2 目标

- 用户注册 / 登录（JWT）
- 每个知识库归属于一个用户
- 知识库可以分享给其他用户（三种角色：viewer / editor / owner）
- Streamlit 和 FastAPI 两条入口都走同一套权限检查
- 已有的单用户数据迁移到一个默认 admin 账户

### 1.3 数据库变更

在 `src/database.py` 的 `init_db()` 中追加以下建表语句：

```sql
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_permissions (
    kb_id   TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role    TEXT NOT NULL DEFAULT 'viewer',
    PRIMARY KEY (kb_id, user_id),
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

在 `knowledge_bases` 表上加一列（用 try/except 包住，已存在则跳过）：

```python
try:
    conn.execute("ALTER TABLE knowledge_bases ADD COLUMN owner_id TEXT REFERENCES users(id)")
except Exception:
    pass
```

**迁移逻辑**：在 `init_db()` 里，建表之后检查 `users` 表是否为空。如果为空，创建默认管理员：

```python
if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
    _create_default_admin(conn)
```

`_create_default_admin(conn)` 的逻辑：
- 用户名 = `os.getenv("ADMIN_USERNAME", "admin")`
- 密码 = `os.getenv("ADMIN_PASSWORD", "changeme")`（bcrypt 哈希后存入）
- role = `"admin"`
- 把所有 `owner_id IS NULL` 的知识库的 `owner_id` 设为 admin 的 id，并在 `kb_permissions` 里插入 `(kb_id, admin_id, 'owner')`。

### 1.4 新文件

```
src/auth/
  __init__.py
  service.py      # 注册 / 登录 / 验证 / 密码哈希
  permissions.py  # 权限检查
  deps.py         # FastAPI Depends + Streamlit session 检查
```

#### `src/auth/service.py`

```python
import bcrypt
import jwt
import os
import uuid
from datetime import datetime, timedelta
from src.database import get_connection

JWT_SECRET = os.getenv("JWT_SECRET", "datalens-dev-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_token(user_id: str, username: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def register(username: str, password: str) -> dict:
    """注册新用户。用户名已存在时抛 ValueError。"""
    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        raise ValueError(f"用户名已存在: {username}")
    user_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, username, hash_password(password), "user", now),
    )
    conn.commit()
    conn.close()
    return {"id": user_id, "username": username, "role": "user"}

def login(username: str, password: str) -> dict | None:
    """验证用户名密码，成功返回 token + 用户信息，失败返回 None。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    token = create_token(row["id"], row["username"], row["role"])
    return {"token": token, "user": {"id": row["id"], "username": row["username"], "role": row["role"]}}
```

#### `src/auth/permissions.py`

```python
from src.database import get_connection

def check_kb_access(user_id: str, kb_id: str, required: str = "viewer") -> bool:
    """检查 user 对 kb 是否有 required 级别的权限。
    角色层级：owner > editor > viewer。admin 全局角色自动拥有所有库的 owner 权限。
    """
    conn = get_connection()
    user = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return False
    if user["role"] == "admin":
        conn.close()
        return True
    kb = conn.execute("SELECT owner_id FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
    if not kb:
        conn.close()
        return False
    if kb["owner_id"] == user_id:
        conn.close()
        return True
    perm = conn.execute(
        "SELECT role FROM kb_permissions WHERE kb_id = ? AND user_id = ?", (kb_id, user_id)
    ).fetchone()
    conn.close()
    if not perm:
        return False
    hierarchy = {"viewer": 0, "editor": 1, "owner": 2}
    return hierarchy[perm["role"]] >= hierarchy[required]

def list_accessible_kbs(user_id: str) -> list[dict]:
    """列出 user 有权限访问的所有知识库。"""
    conn = get_connection()
    user = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return []
    if user["role"] == "admin":
        rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT kb.* FROM knowledge_bases kb
            LEFT JOIN kb_permissions p ON kb.id = p.kb_id
            WHERE kb.owner_id = ? OR p.user_id = ?
            ORDER BY kb.created_at DESC
        """, (user_id, user_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def share_kb(kb_id: str, target_user_id: str, role: str):
    """把知识库分享给另一个用户。role 只能是 viewer / editor / owner。"""
    assert role in ("viewer", "editor", "owner")
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO kb_permissions (kb_id, user_id, role) VALUES (?, ?, ?)",
        (kb_id, target_user_id, role),
    )
    conn.commit()
    conn.close()
```

#### `src/auth/deps.py`

```python
from fastapi import HTTPException, Header
from src.auth.service import decode_token
from src.auth.permissions import check_kb_access

def get_current_user(authorization: str = Header(...)) -> dict:
    """FastAPI 依赖：从 Authorization: Bearer <token> 提取用户。"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="无效的认证头")
    token = authorization[7:]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="token 无效或已过期")
    return payload

def require_kb_access(kb_id: str, user: dict, required: str = "viewer"):
    """通用权限检查，不通过则抛 403。"""
    if not check_kb_access(user["sub"], kb_id, required):
        raise HTTPException(status_code=403, detail=f"没有该知识库的 {required} 权限")
```

### 1.5 API 变更

在 `src/api/web_api.py` 中新增路由：

```python
class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6)

class LoginRequest(BaseModel):
    username: str
    password: str

class ShareRequest(BaseModel):
    target_username: str
    role: str

@router.post("/auth/register")
def register(req: RegisterRequest): ...

@router.post("/auth/login")
def login(req: LoginRequest): ...

@router.get("/auth/me")
def me(user: dict = Depends(get_current_user)): ...

@router.post("/kbs/{kb_id}/share")
def share_kb_endpoint(kb_id: str, req: ShareRequest, user: dict = Depends(get_current_user)):
    ...
```

给现有端点加权限的方式：每个端点函数签名加 `user: dict = Depends(get_current_user)`，内部调用 `require_kb_access(kb_id, user, required_role)`。

| 端点 | 权限要求 |
|---|---|
| `GET /api/kbs` | 登录即可（只返回有权限的库） |
| `GET /api/kbs/{kb_id}/docs` | viewer |
| `POST /api/ask` | viewer |
| `POST /api/ask/stream` | viewer |
| `POST /api/kbs/{kb_id}/upload` | editor |
| `POST /api/kbs/{kb_id}/reindex` | editor |
| `DELETE /api/docs/{doc_id}` | editor |
| `GET /api/answer_log` | viewer |
| `GET /api/eval/history` | viewer |

### 1.6 Streamlit 变更

在 `app.py` 最顶部（`st.set_page_config` 之后）加登录检查：

```python
if "user" not in st.session_state:
    st.title("登录 DataLens")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        from src.auth.service import login
        result = login(username, password)
        if result:
            st.session_state.user = result["user"]
            st.session_state.token = result["token"]
            st.rerun()
        else:
            st.error("用户名或密码错误")
    st.stop()
```

在 `pages/1` 中把 `list_kbs()` 改为 `list_accessible_kbs(st.session_state.user["id"])`。
同理修改 `pages/2` `pages/3` `pages/4` 中所有 `list_kbs()` / `get_kb()` 调用。

### 1.7 依赖

在 `requirements.txt` 中添加：

```
bcrypt>=4.0
PyJWT>=2.8
```

### 1.8 测试

新增 `tests/test_auth.py`，至少覆盖：
1. 注册、登录、拿到 token、decode token 拿到 user_id
2. 重复注册用户名抛 ValueError
3. 密码错误 login 返回 None
4. `check_kb_access`：owner 可以 / viewer 不能 upload / 非授权用户不能访问
5. admin 可以访问任何库
6. `list_accessible_kbs` 只返回自己创建的和被分享的
7. 迁移逻辑：空库启动自动创建 admin、已有 KB 的 owner_id 被赋值

### 1.9 验收标准

- 未登录访问 `GET /api/kbs` 返回 401
- 用 admin 登录后可以访问所有库
- 注册新用户新建库另一个用户看不到分享后能看到
- Streamlit 打开时先显示登录页登录后正常使用

---

## 二、Contextual Retrieval

### 2.1 原理

入库前用 LLM 给每个 chunk 生成一句"这个片段在整篇文档里的上下文是什么"，拼在 chunk 前面再嵌入。

### 2.2 现状

`chunker.py` 的标题树模式会拼父路径但只是标题文字不是语义描述。对 PDF / OCR 场景标题路径经常为空。

### 2.3 实现

新建 `src/contextualizer.py`：

```python
"""
Contextual Retrieval：给每个 chunk 生成一句文档级上下文拼在 chunk 前再嵌入。
成本：入库时每个 chunk 多一次 LLM 调用。检索阶段零额外开销。
"""
import sys
from openai import OpenAI
from src.config import (
    CONTEXTUAL_RETRIEVAL_ENABLED,
    CONTEXTUAL_MODEL,
    CONTEXTUAL_DOCUMENT_MAX_CHARS,
)

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from src.config import LLM_API_KEY, LLM_BASE_URL
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client

SYSTEM_PROMPT = (
    "你是一个文档上下文生成器。我会给你一份文档的内容片段和这篇文档的完整文本（可能被截断）。\n"
    "你的任务：用 1-2 句话描述这个片段在整篇文档中的位置和语境。\n"
    "规则：只描述片段在文档中的位置和主题，不要回答片段中的问题。\n"
    "不要引入文档中没有的信息。只输出描述文字。\n"
    "如果片段本身已经自包含输出：此片段自成一体，无需额外上下文。"
)

def generate_context(chunk_text: str, full_document: str) -> str:
    """给一个 chunk 生成上下文描述。失败时返回空字符串。"""
    if not CONTEXTUAL_RETRIEVAL_ENABLED:
        return ""
    truncated_doc = full_document[:CONTEXTUAL_DOCUMENT_MAX_CHARS]
    user_prompt = f"<document>\n{truncated_doc}\n</document>\n\n<chunk>\n{chunk_text}\n</chunk>\n\n上下文描述："
    try:
        response = _get_client().chat.completions.create(
            model=CONTEXTUAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        result = response.choices[0].message.content.strip()
        if "自成一体" in result:
            return ""
        return result
    except Exception as e:
        print(f"[Contextualizer] 生成上下文失败: {e}", file=sys.stderr)
        return ""

def contextualize_chunks(chunks: list[dict], full_document: str) -> list[dict]:
    """批量处理。修改每个 chunk 的 content 在前面拼上上下文前缀。失败时某个 chunk 不加前缀。"""
    for chunk in chunks:
        context = generate_context(chunk["content"], full_document)
        if context:
            chunk["content"] = f"[{context}] {chunk['content']}"
    return chunks
```

### 2.4 配置

```python
CONTEXTUAL_RETRIEVAL_ENABLED = os.getenv("CONTEXTUAL_RETRIEVAL_ENABLED", "false").lower() == "true"
CONTEXTUAL_MODEL = os.getenv("CONTEXTUAL_MODEL", "deepseek-chat")
CONTEXTUAL_DOCUMENT_MAX_CHARS = int(os.getenv("CONTEXTUAL_DOCUMENT_MAX_CHARS", "8000"))
```

### 2.5 接入点

在 `src/parser.py` 的 `parse_file()` 返回后、`chunker.chunk_parsed()` 后、`vector_store.add_chunks()` 前：

```python
if CONTEXTUAL_RETRIEVAL_ENABLED:
    from src.contextualizer import contextualize_chunks
    chunks = contextualize_chunks(chunks, full_document_text)
```

如果 `parse_file()` 当前没有返回完整原文在返回值里加一个 `"full_text"` 字段。

### 2.6 测试

新增 `tests/test_contextualizer.py`：
1. `CONTEXTUAL_RETRIEVAL_ENABLED=false` 时 `generate_context()` 返回空串
2. mock LLM 返回正常描述时 chunk content 前面出现 `[描述] `
3. mock LLM 返回"自成一体"时 chunk content 不变
4. mock LLM 抛异常时 chunk content 不变

---

## 三、Parent-Child Retrieval

### 3.1 原理

小块用于嵌入和检索，命中后返回它所在的上下文邻域（相邻 chunk 合并）给 LLM。

### 3.2 实现方案

不需要数据库变更。精排命中的 chunk 按 source 和 chunk_index 从向量库拉取相邻 chunk 合并后作为上下文。

新建 `src/context_expander.py`：

```python
"""
Parent-Child Retrieval：精排命中小块后拉取相邻 chunk 合并成大块给 LLM。
"""
from src.config import CONTEXT_EXPANSION_ENABLED, CONTEXT_EXPAND_NEIGHBORS, CONTEXT_EXPAND_MAX_CHARS

def expand_contexts(hits: list[dict], kb_id: str) -> list[dict]:
    """扩展每个命中 chunk 的上下文。相邻命中合并去重。"""
    if not CONTEXT_EXPANSION_ENABLED or not hits:
        return hits
    from src.vector_store import get_chunks_by_source_range
    expanded = []
    seen_ranges = set()
    for hit in hits:
        source = hit["source"]
        idx = hit["chunk_index"]
        start = max(0, idx - CONTEXT_EXPAND_NEIGHBORS)
        end = idx + CONTEXT_EXPAND_NEIGHBORS
        key = (source, start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        neighbors = get_chunks_by_source_range(kb_id, source, start, end)
        merged_text = "\n".join(n["content"] for n in neighbors)
        if len(merged_text) > CONTEXT_EXPAND_MAX_CHARS:
            merged_text = merged_text[:CONTEXT_EXPAND_MAX_CHARS] + "..."
        expanded.append({**hit, "content": merged_text})
    return expanded
```

### 3.3 `vector_store.py` 新增函数

```python
def get_chunks_by_source_range(kb_id: str, source: str, start_idx: int, end_idx: int) -> list[dict]:
    """按 source 和 chunk_index 范围获取 chunk 列表（按 chunk_index 升序）。"""
    client = _get_client()
    collection = client.get_collection(_collection_name(kb_id))
    results = collection.get(
        where={"$and": [
            {"source": {"$eq": source}},
            {"chunk_index": {"$gte": start_idx}},
            {"chunk_index": {"$lte": end_idx}},
        ]},
        include=["documents", "metadatas"],
    )
    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({
            "content": results["documents"][i],
            "source": results["metadatas"][i].get("source", ""),
            "page": results["metadatas"][i].get("page", 0),
            "chunk_index": results["metadatas"][i].get("chunk_index", 0),
        })
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks
```

### 3.4 配置

```python
CONTEXT_EXPANSION_ENABLED = os.getenv("CONTEXT_EXPANSION_ENABLED", "false").lower() == "true"
CONTEXT_EXPAND_NEIGHBORS = int(os.getenv("CONTEXT_EXPAND_NEIGHBORS", "2"))
CONTEXT_EXPAND_MAX_CHARS = int(os.getenv("CONTEXT_EXPAND_MAX_CHARS", "2000"))
```

### 3.5 接入点

在 `src/rag_qa.py` 中精排返回 top-k 后拼上下文给 LLM 之前：

```python
if CONTEXT_EXPANSION_ENABLED:
    from src.context_expander import expand_contexts
    contexts = expand_contexts(contexts, kb_id)
```

注意：引用来源的 snippets 应该显示精排命中的原始 chunk 内容不是合并后的长文本。所以在扩展前先把原始片段存下来。

### 3.6 测试

新增 `tests/test_context_expander.py`：
1. 关闭时原样返回
2. mock 返回 5 个 chunk 合并成一个字符串
3. 两个 hit 同一 source 相邻 chunk_index 第二个不重复扩展
4. 合并后超过 max_chars 截断

---

## 四、Multi-Query Expansion

### 4.1 原理

把一个问题改写成 3-5 个不同角度的表述，每个都去检索，合并去重后送精排。

### 4.2 实现

新建 `src/query_expand.py`：

```python
"""
查询扩展：把一个检索 query 变成多个等价表述分别检索后合并候选。
与 query_rewrite 的关系：query_rewrite 解决指代不完整先于本模块执行。
query_expand 解决表述单一在 rewrite 之后执行。
"""
import sys
from openai import OpenAI
from src.config import QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_COUNT, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client

EXPAND_SYSTEM_PROMPT = (
    "你是一个检索查询扩展器。我会给你一个用户问题你需要把它改写成 {count} 个不同角度的等价检索查询。\n"
    "规则：每个查询都是独立完整的检索语句。每个查询从不同角度表述同一个信息需求。\n"
    "不要回答问题不要解释不要输出编号。每行一个查询只输出查询文本。\n"
    "如果问题很短或已经很通用输出原问题 {count} 次。"
)

def expand_query(query: str) -> list[str]:
    """扩展一个 query 为多个等价查询。失败或关闭时返回 [query]。"""
    if not QUERY_EXPANSION_ENABLED:
        return [query]
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXPAND_SYSTEM_PROMPT.format(count=QUERY_EXPANSION_COUNT)},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        lines = [line.strip() for line in response.choices[0].message.content.strip().split("\n") if line.strip()]
        queries = list(dict.fromkeys([query] + lines))
        return queries[:QUERY_EXPANSION_COUNT]
    except Exception as e:
        print(f"[QueryExpand] 扩展失败退回原始 query: {e}", file=sys.stderr)
        return [query]
```

### 4.3 检索流程变更

在 `src/rag_qa.py` 的检索阶段改为：

```python
retrieval_query = rewrite_query(query, history)
expanded_queries = expand_query(retrieval_query)
all_candidates = []
for eq in expanded_queries:
    candidates = retriever.search(kb_id, eq, top_k=RETRIEVE_CANDIDATES or TOP_K_RETRIEVE * 10)
    all_candidates.extend(candidates)
seen = set()
deduped = []
for c in all_candidates:
    key = c["content"][:100]
    if key not in seen:
        seen.add(key)
        deduped.append(c)
```

注意：`retrieval_query` 返回给前端的仍然是 `rewrite_query()` 的结果单个字符串。

### 4.4 配置

```python
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true"
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
```

### 4.5 测试

新增 `tests/test_query_expand.py`：
1. 关闭时返回 `[query]`
2. mock LLM 返回 3 行返回 4 条含原始 query
3. mock LLM 抛异常返回 `[query]`
4. mock 返回的某条与原始 query 相同去重后不重复

---

## 五、CRAG 生成后自检

### 5.1 原理

LLM 生成答案后让模型判断"这个答案是否真的基于给出的上下文"。如果不 grounded 用更严格的 prompt 重试。

### 5.2 实现

新建 `src/answer_verifier.py`：

```python
"""
CRAG：生成后自检验证答案是否 grounded。
失败即放行，验证的目的是提高质量不能反过来阻断回答。
"""
import sys
from openai import OpenAI
from src.config import ANSWER_VERIFY_ENABLED, ANSWER_VERIFY_THRESHOLD, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client

VERIFY_SYSTEM_PROMPT = (
    "你是一个 RAG 答案验证器。我会给你一段上下文和一个基于该上下文生成的答案。\n"
    "判断答案是否完全基于上下文中的信息。评分标准 1-5：\n"
    "5 = 每个陈述都能在上下文中找到直接依据\n"
    "4 = 基本基于上下文有少量合理推断\n"
    "3 = 部分基于上下文部分内容来源不明\n"
    "2 = 有明显的编造或与上下文矛盾\n"
    "1 = 完全脱离上下文\n"
    "只输出一个数字 1-5 不要解释。"
)

def verify_answer(answer: str, contexts: list[str]) -> int:
    """返回 grounded 分数 1-5。失败返回 5（默认放行）。"""
    if not ANSWER_VERIFY_ENABLED or not contexts:
        return 5
    context_text = "\n---\n".join(contexts[:5])[:4000]
    user_prompt = f"<context>\n{context_text}\n</context>\n\n<answer>\n{answer}\n</answer>\n\n分数："
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        score = int(response.choices[0].message.content.strip().rstrip("."))
        return max(1, min(5, score))
    except Exception as e:
        print(f"[AnswerVerifier] 验证失败默认放行: {e}", file=sys.stderr)
        return 5
```

### 5.3 接入点

在 `src/rag_qa.py` 的生成函数中 LLM 返回答案后：

```python
if ANSWER_VERIFY_ENABLED and contexts:
    from src.answer_verifier import verify_answer
    score = verify_answer(answer, [c["content"] for c in contexts])
    if score < ANSWER_VERIFY_THRESHOLD:
        # 用更严格的 prompt 重试一次
        pass
```

### 5.4 配置

```python
ANSWER_VERIFY_ENABLED = os.getenv("ANSWER_VERIFY_ENABLED", "false").lower() == "true"
ANSWER_VERIFY_THRESHOLD = int(os.getenv("ANSWER_VERIFY_THRESHOLD", "3"))
```

### 5.5 测试

新增 `tests/test_answer_verifier.py`：
1. 关闭时返回 5
2. mock LLM 返回 "4" 返回 4
3. mock LLM 返回 "2" 返回 2
4. mock LLM 抛异常返回 5

---

## 六、高相似直接返回

### 6.1 原理

当精排最高分超过阈值（如 0.92）跳过 LLM 生成直接返回该 chunk 的原文。

### 6.2 实现

在 `src/rag_qa.py` 中精排返回 top-k 后调用 LLM 生成之前：

```python
if DIRECT_RETURN_ENABLED and contexts:
    from src.answer_gate import top_score
    if top_score(contexts) >= DIRECT_RETURN_THRESHOLD:
        best = contexts[0]
        answer = f"根据知识库中的以下内容：\n\n{best['content']}"
        # 记录 answer_log 标记 direct_return=True
        # 跳过 LLM 调用直接 return
```

### 6.3 配置

```python
DIRECT_RETURN_ENABLED = os.getenv("DIRECT_RETURN_ENABLED", "false").lower() == "true"
DIRECT_RETURN_THRESHOLD = float(os.getenv("DIRECT_RETURN_THRESHOLD", "0.92"))
```

### 6.4 测试

新增 `tests/test_direct_return.py`：
1. 关闭时不触发
2. mock 精排分数 0.95 大于阈值时不调 LLM 返回 chunk 内容
3. mock 精排分数 0.5 小于阈值时正常走 LLM

---

## 七、Adaptive Routing

### 7.1 原理

在检索之前判断问题类型：chitchat 闲聊不走 RAG / simple 单跳正常 RAG / multi-hop 多跳拆分检索。

### 7.2 实现

新建 `src/query_router.py`：

```python
"""
查询路由：判断问题类型决定走哪条检索路径。
三层路由：规则引擎（零成本） -> LLM 分类 -> 多跳分解。
门控 answer_gate 仍然保留在检索后作为最后一道防线。
"""
import re
from src.config import ADAPTIVE_ROUTING_ENABLED

_CHITCHAT_PATTERNS = [
    re.compile(r"^(你好|hi|hello|嗨|哈喽)", re.I),
    re.compile(r"^(谢谢|thanks|thank you)", re.I),
    re.compile(r"(天气|时间|日期|几点)"),
    re.compile(r"^(你是谁|你叫什么|你能做什么)"),
    re.compile(r"^(帮我写|帮我画|帮我做)(?!(.*(?:文档|知识库|资料)))", re.I),
]

def classify_query(query: str) -> str:
    """返回 chitchat / simple / multi-hop。规则引擎先行不确定时用 LLM 分类。"""
    if not ADAPTIVE_ROUTING_ENABLED:
        return "simple"
    for pattern in _CHITCHAT_PATTERNS:
        if pattern.search(query):
            return "chitchat"
    return _llm_classify(query)

def _llm_classify(query: str) -> str:
    from src.config import LLM_API_KEY, LLM_BASE_URL
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "判断这个问题的类型只输出一个词：\n"
                    "chitchat = 闲聊常识与知识库无关\n"
                    "simple = 单一事实查询一次检索就能回答\n"
                    "multi-hop = 需要组合两个或以上事实才能回答\n"
                    "只输出 chitchat simple multi-hop 之一。"
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().lower()
        if result in ("chitchat", "simple", "multi-hop"):
            return result
        return "simple"
    except Exception:
        return "simple"

def decompose_query(query: str) -> list[str]:
    """把多跳问题拆成子问题列表。失败返回 [query]。"""
    try:
        from openai import OpenAI
        from src.config import LLM_API_KEY, LLM_BASE_URL
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "把一个多跳问题拆成按顺序回答的子问题列表。\n"
                    "每个子问题独立可检索后一个可以依赖前一个的答案。\n"
                    "每行一个子问题不要编号不要解释。如果问题不是多跳的原样输出。"
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        lines = [l.strip() for l in response.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines if lines else [query]
    except Exception:
        return [query]
```

### 7.3 接入点

在 `src/rag_qa.py` 的 `rag_query()` 中 `rewrite_query()` 之后：

```python
from src.query_router import classify_query, decompose_query

query_type = classify_query(retrieval_query)

if query_type == "chitchat":
    contexts = []
    grounded = False
elif query_type == "multi-hop":
    sub_queries = decompose_query(retrieval_query)
    all_contexts = []
    for sq in sub_queries:
        hits = retriever.search(kb_id, sq, top_k=TOP_K_RETRIEVE)
        all_contexts.extend(hits)
    contexts = _deduplicate(all_contexts)[:TOP_K_RETRIEVE]
    grounded = True
else:
    contexts = retriever.search(kb_id, retrieval_query, top_k=TOP_K_RETRIEVE)
```

### 7.4 配置

```python
ADAPTIVE_ROUTING_ENABLED = os.getenv("ADAPTIVE_ROUTING_ENABLED", "false").lower() == "true"
```

### 7.5 测试

新增 `tests/test_query_router.py`：
1. 关闭时返回 simple
2. "你好" 返回 chitchat 规则引擎命中
3. "混合检索是怎么做的" 返回 simple（mock LLM）
4. "A 的作者和B 的作者是什么关系" 返回 multi-hop（mock LLM）
5. `decompose_query` mock 返回 2 行返回 2 个子问题
6. `decompose_query` mock 抛异常返回 `[query]`

---

## 八、内容去重

### 8.1 现状

`documents` 表没有 `content_hash` 字段。上传同一份文件两次会重复入库。

### 8.2 目标

上传时计算 SHA-256 如果同一个 KB 里已有相同 hash 的文件提示"文件已存在"跳过。

### 8.3 数据库变更

```python
try:
    conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT DEFAULT ''")
except Exception:
    pass
```

### 8.4 实现

在 `src/vector_store.py` 中：

```python
import hashlib

def compute_content_hash(chunks: list[dict]) -> str:
    combined = "\n".join(c["content"] for c in chunks)
    return hashlib.sha256(combined.encode()).hexdigest()

def check_duplicate(kb_id: str, content_hash: str) -> bool:
    from src.database import get_connection
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE kb_id = ? AND content_hash = ? AND status = 'ready'",
        (kb_id, content_hash),
    ).fetchone()
    conn.close()
    return row[0] > 0
```

调用点：在 upload 逻辑中 `parse_file()` 之后：

```python
content_hash = compute_content_hash(chunks)
if check_duplicate(kb_id, content_hash):
    raise HTTPException(409, "内容与已有文件重复跳过入库")
```

同时在 `add_document()` 时把 `content_hash` 写入 documents 表。

### 8.5 测试

新增 `tests/test_content_dedup.py`：
1. 上传同一内容 `check_duplicate` 返回 True
2. 上传不同内容返回 False

---

## 九、实施顺序

| 顺序 | 模块 | 依赖 | 风险 | 工作量 |
|:---:|---|---|:---:|:---:|
| 1 | 多用户与权限 | 无 | 中 | 大 |
| 2 | 内容去重 | 权限 | 低 | 小 |
| 3 | 高相似直接返回 | 无 | 低 | 小 |
| 4 | Contextual Retrieval | 无 | 低 | 小 |
| 5 | Multi-Query Expansion | 无 | 低 | 小 |
| 6 | CRAG | 无 | 低 | 小 |
| 7 | Adaptive Routing | 无 | 中 | 中 |
| 8 | Parent-Child Retrieval | 无 | 低 | 中 |

每个模块单独一个 git commit。出问题时精确回退。

每完成一个模块后的验证步骤：
1. `pytest tests/ -x` 全量测试通过
2. 启动 API 服务 `GET /api/health` 返回 200
3. 启动 Streamlit 登录建库上传问答引用正常
4. 用 `run_eval_experiment.py` 跑一次评估确认指标没有意外下降

---

## 十、测试文件清单

| 文件 | 测什么 | 最少用例数 |
|---|---|:---:|
| `tests/test_auth.py` | 注册登录token权限迁移 | 10 |
| `tests/test_contextualizer.py` | 上下文生成降级 | 4 |
| `tests/test_context_expander.py` | 邻域扩展去重截断 | 4 |
| `tests/test_query_expand.py` | 查询扩展降级 | 4 |
| `tests/test_answer_verifier.py` | CRAG验证降级 | 4 |
| `tests/test_direct_return.py` | 高相似直接返回 | 3 |
| `tests/test_query_router.py` | 路由分类多跳分解降级 | 6 |
| `tests/test_content_dedup.py` | 内容去重 | 2 |

所有 mock 用 `unittest.mock.patch` 或 `pytest-mock` 不发真实 API 请求。
