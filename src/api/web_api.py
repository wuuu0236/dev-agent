"""
Web API —— 给 React 前端（frontend/）提供 REST 端点

设计原则：
  · 这一层**只做协议转换**（HTTP ↔ 现有模块调用），不含任何业务逻辑。
    检索、门控、解析、入库全部复用 Streamlit 已有的模块，保证两条入口行为一致。
  · 函数内部 import：任何模块在无 API key 环境下也必须能 import（CI 守卫），
    而本文件只被 API 服务加载，运行时 import 不影响那条守卫。

端点一览（除 /health 与 /auth/* 外**全部需要登录**，括号内是该端点要求的库权限）：
  GET    /api/health                  服务可用性与后端连通状态（免登录）
  POST   /api/auth/register           注册（免登录）
  POST   /api/auth/login              登录，换取 JWT（免登录）
  GET    /api/auth/me                 当前登录用户信息
  POST   /api/kbs/{kb_id}/share       把知识库分享给他人（owner）
  GET    /api/kbs                     知识库列表（只返回当前用户有权限的）
  GET    /api/kbs/{kb_id}/docs        某知识库的文档列表（viewer）
  POST   /api/ask                     RAG 问答（改写 → 检索 → 精排 → 门控 → 生成 → 引用）（viewer）
  POST   /api/ask/stream              同上，但答案用 SSE 逐块推送（前端打字机效果）（viewer）
  POST   /api/feedback                给某条回答打反馈：👍/👎 + 可选备注（viewer）
  GET    /api/kbs/{kb_id}/gaps        待补知识清单：👎 汇成的待办（viewer）
  POST   /api/gaps/{gap_id}           处理待补事项：done / ignored / open（editor）
  POST   /api/kbs/{kb_id}/upload      上传并入库（解析 → 切块 → 嵌入）（editor）
  POST   /api/kbs/{kb_id}/reindex     按当前参数重建索引（editor）
  DELETE /api/docs/{doc_id}           删除文档（SQLite + Chroma 双清）（editor）
  GET    /api/answer_log              问答日志 + 缺口分类（viewer）
  GET    /api/eval/history            RAGAS 评估历史存档（登录即可）

鉴权：`Authorization: Bearer <token>`，token 由 /auth/login 签发，有效期见
config.JWT_EXPIRE_HOURS。权限规则本身在 src/auth/permissions.py，这一层只做转换。
"""

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.auth.deps import get_current_user, require_kb_access, user_id_of

router = APIRouter(prefix="/api", tags=["web"])


# ================================================================
# 请求/响应模型
# ================================================================

class AskRequest(BaseModel):
    """RAG 问答请求"""
    kb_id: str = Field(description="知识库 ID")
    question: str = Field(description="用户问题")
    top_k: int = Field(default=5, description="进入上下文的 chunk 数")
    history: list[dict] = Field(
        default_factory=list,
        description="此条问题之前的对话历史（不含当前问题），格式 [{role, content}]，最近的在末尾",
    )


class AskResponse(BaseModel):
    """RAG 问答响应：answer + 引用 + 现场元信息（供前端展示门控/缓存/精排分）"""
    answer: str
    grounded: bool = Field(description="是否通过了检索质量门控（False = 如实说没找到）")
    gate_score: float | None = Field(description="本次检索最高精排分（缓存命中时为 None）")
    retrieval_query: str = Field(description="实际用于检索的 query（追问时是消解后的自包含问句）")
    sources: list[dict] = Field(description="引用来源，含命中原文片段")
    log_id: int | None = None


# ================================================================
# 端点
# ================================================================

@router.get("/health", summary="健康检查")
def health():
    return {"status": "ok", "service": "DataLens Web API"}


# ================================================================
# 认证（注册 / 登录 / 当前用户 / 分享）
# ================================================================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32, description="用户名，2-32 字符")
    password: str = Field(min_length=6, description="密码，至少 6 位")


class LoginRequest(BaseModel):
    username: str
    password: str


class ShareRequest(BaseModel):
    target_username: str = Field(description="分享给谁（用户名）")
    role: str = Field(default="viewer", description="viewer / editor / owner 之一")


@router.post("/auth/register", summary="注册")
def register(req: RegisterRequest):
    from src.auth.service import register as _register

    try:
        return _register(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/auth/login", summary="登录（换取 JWT）")
def login(req: LoginRequest):
    from src.auth.service import login as _login

    result = _login(req.username, req.password)
    if not result:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return result


@router.get("/auth/me", summary="当前登录用户")
def me(user: dict = Depends(get_current_user)):
    from src.auth.service import get_user

    info = get_user(user_id_of(user))
    if not info:
        # token 本身合法、但账号已被删 —— 当作未登录处理，
        # 否则一个已删除的账号能在 token 过期前继续访问（最长期限 = JWT_EXPIRE_HOURS）
        raise HTTPException(status_code=401, detail="用户不存在")
    return info


@router.post("/kbs/{kb_id}/share", summary="分享知识库")
def share_kb_endpoint(kb_id: str, req: ShareRequest, user: dict = Depends(get_current_user)):
    from src.auth.permissions import find_user_by_username, share_kb

    require_kb_access(kb_id, user, "owner")  # 只有 owner 能授权给别人
    target = find_user_by_username(req.target_username)
    if not target:
        raise HTTPException(status_code=404, detail=f"用户不存在: {req.target_username}")
    try:
        share_kb(kb_id, target["id"], req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "kb_id": kb_id, "user": target["username"], "role": req.role}


# ================================================================
# 知识库
# ================================================================

@router.get("/kbs", summary="知识库列表")
def get_kbs(user: dict = Depends(get_current_user)):
    from src.auth.permissions import list_accessible_kbs
    from src.database import get_kb_stats

    result = []
    for kb in list_accessible_kbs(user_id_of(user)):
        stats = get_kb_stats(kb["id"])
        try:
            from src.vector_store import collection_count

            indexed = collection_count(kb["id"])
        except Exception:
            indexed = 0  # 集合尚未创建（新库没传过文档），不算错误
        result.append({**kb, **stats,
                       "doc_count": stats.get("doc_count", 0),
                       "total_chunks": stats.get("total_chunks", 0),
                       "indexed_chunks": indexed})
    return result


@router.get("/kbs/{kb_id}/docs", summary="某知识库的文档列表")
def get_docs(kb_id: str, user: dict = Depends(get_current_user)):
    from src.database import get_kb, get_kb_stats, list_documents

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")
    require_kb_access(kb_id, user, "viewer")
    return {"kb_id": kb_id, "stats": get_kb_stats(kb_id), "docs": list_documents(kb_id)}


@router.post("/ask", response_model=AskResponse, summary="RAG 问答")
def ask(request: AskRequest, user: dict = Depends(get_current_user)):
    """走与 Streamlit 完全相同的 rag_query 链路，返回答案 + 引用 + 门控信息。"""
    from src.answer_gate import top_score
    from src.answer_log import get_answer
    from src.database import get_kb
    from src.rag_qa import rag_query

    if not get_kb(request.kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {request.kb_id}")
    require_kb_access(request.kb_id, user, "viewer")

    history = request.history or None
    result = rag_query(request.kb_id, request.question,
                       top_k=request.top_k, history=history)

    # 门控未通过时 contexts 已被清空，此时从本次落下的日志里取回门控分数——
    # 前端要展示「最高精排分 0.28 但没过阈」，这个数只能这么拿。
    gate_score = top_score(result["contexts"]) if result["contexts"] else None
    if gate_score is None and result.get("log_id"):
        log = get_answer(result["log_id"])
        gate_score = log.get("gate_score") if log else None

    sources = [
        {"n": i + 1, "file": s["source"], "page": s.get("page", 0),
         "snippets": s.get("snippets", []), "type": s.get("type", "text")}
        for i, s in enumerate(result["sources"])
    ]

    return AskResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        gate_score=gate_score,
        retrieval_query=result["retrieval_query"],
        sources=sources,
        log_id=result.get("log_id"),
    )


def _sse(event: str, data: dict) -> str:
    """拼一条 SSE 消息。JSON 会把换行转义成 \\n，因此 data 一定是单行，符合 SSE 规范。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask/stream", summary="RAG 问答（SSE 流式，答案逐块推送）")
def ask_stream(request: AskRequest, user: dict = Depends(get_current_user)):
    """与 `/api/ask` 完全同一条链路，只是答案边生成边推送，用于前端打字机效果。

    事件序列：
      `meta`  —— 引用来源 / 实际检索 query / 门控分数。检索与精排在生成之前就结束了，
                 所以这几个值不必等模型，前端可先把引用骨架渲染出来。
      `delta` —— 答案片段（多条，按顺序拼接）。
      `done`  —— 完整答案 + 本次日志 id（供反馈与后续追问）。
    """
    from src.database import get_kb
    from src.rag_qa import stream_rag_query

    if not get_kb(request.kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {request.kb_id}")
    require_kb_access(request.kb_id, user, "viewer")

    gen, sources, contexts, retrieval_query, log_ref = stream_rag_query(
        request.kb_id, request.question, top_k=request.top_k,
        history=request.history or None,
    )

    payload_sources = [
        {"n": i + 1, "file": s["source"], "page": s.get("page", 0),
         "snippets": s.get("snippets", []), "type": s.get("type", "text")}
        for i, s in enumerate(sources)
    ]

    def iter_events():
        gate_score = log_ref.get("gate_score")
        yield _sse("meta", {
            "grounded": bool(contexts),
            "gate_score": gate_score,
            "retrieval_query": retrieval_query,
            "sources": payload_sources,
        })
        chunks = []
        for chunk in gen:
            chunks.append(chunk)
            yield _sse("delta", {"text": chunk})
        # 生成器跑完后日志才落库，log_id 这时才能取到（缓存命中分支是例外，当时就有）
        yield _sse("done", {
            "answer": "".join(chunks),
            "log_id": log_ref.get("id"),
            "gate_score": gate_score,
        })

    return StreamingResponse(
        iter_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class FeedbackRequest(BaseModel):
    """对某条回答的人工反馈"""
    log_id: int = Field(description="要打反馈的问答日志 id（/api/ask 或 done 事件返回）")
    rating: str | None = Field(default=None, description="'up' / 'down' / null（null = 撤销）")
    comment: str | None = Field(
        default=None,
        description="可选补充说明（'哪里不对'）。只在传了值时才覆盖已有备注，"
                    "避免前端「只点赞没写备注」把之前写的抹掉",
    )


@router.post("/feedback", summary="给某条回答打反馈（👍/👎）")
def feedback(request: FeedbackRequest, user: dict = Depends(get_current_user)):
    """把人工反馈写回 answer_log.rating，并把信号分流到出口（反馈环 P2）。

    rating 只允许 None / 'up' / 'down'，非法值由 answer_log.set_rating 抛错后转 400。
    带反馈的记录不会被容量淘汰清理，因此这些标注是稳定的信号源。

    👎 **不只是记一笔**：它进「待补知识」清单（`GET /kbs/{kb_id}/gaps`），
    且当这条是缓存命中时**自动作废那条缓存**——一个错的答案被缓存后会持续错下去
    （同一个人不会再问第二遍，所以它永远不会被自然纠正）。up / 撤销则收回该条待办。

    权限：入参只有 log_id，必须先查出这条日志属于哪个库才能判权限——
    否则任何登录用户都能给别人的知识库打反馈，直接污染对方的质量信号。
    """
    from src.answer_log import get_answer, set_rating

    log = get_answer(request.log_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"问答记录不存在: {request.log_id}")
    require_kb_access(log["kb_id"], user, "viewer")

    try:
        hit = set_rating(request.log_id, request.rating, request.comment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not hit:
        raise HTTPException(status_code=404, detail=f"问答记录不存在: {request.log_id}")
    return {"ok": True, "log_id": request.log_id, "rating": request.rating}


@router.get("/kbs/{kb_id}/gaps", summary="待补知识清单（👎 汇成的待办）")
def list_kb_gaps(kb_id: str, status: str = "open",
                 user: dict = Depends(get_current_user)):
    """列出该库的待补事项，每条**带原问题的现场**（gate_score / 备注 / 是否缓存命中）。

    为什么把现场一起返回：处理的人要判断"该补文档还是该调检索"（分界线见
    `answer_log.gap_stats`），只给一个问题标题的话还得回头翻日志——多一步就没人做。

    `status` 传空串 = 看全部（含已处理的）。
    """
    from src.answer_log import count_gaps, list_gaps
    from src.database import get_kb

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")
    require_kb_access(kb_id, user, "viewer")
    return {
        "kb_id": kb_id,
        "open_count": count_gaps(kb_id, "open"),
        "items": list_gaps(kb_id, status or None),
    }


class GapStatusRequest(BaseModel):
    """待补事项的处理结果"""
    status: str = Field(
        description="'done'（已解决，通常是补了原文并重建索引）/ "
                    "'ignored'（确认不用管）/ 'open'（重开）"
    )


@router.post("/gaps/{gap_id}", summary="处理待补事项：已解决 / 忽略 / 重开")
def update_gap(gap_id: int, request: GapStatusRequest,
               user: dict = Depends(get_current_user)):
    """把一条待补事项标记成已处理，同时更新 `answer_log.resolved`。

    权限比打反馈高一档（feedback 是 viewer，这里是 editor）：`done` 的语义通常是
    "我已经补了原文并重建了索引"——那是 editor 才能做的动作。让只能提问的人把待办
    标成已解决，清单就会失真，而**失真的清单比没有清单更糟**：它会让人以为处理过了。
    """
    from src.answer_log import get_gap, set_gap_status

    gap = get_gap(gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail=f"待补事项不存在: {gap_id}")
    require_kb_access(gap["kb_id"], user, "editor")

    try:
        hit = set_gap_status(gap_id, request.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not hit:
        raise HTTPException(status_code=404, detail=f"待补事项不存在: {gap_id}")
    return {"ok": True, "gap_id": gap_id, "status": request.status}


@router.post("/kbs/{kb_id}/upload", summary="上传文档并入库")
async def upload(kb_id: str, file: UploadFile = File(...),
                 user: dict = Depends(get_current_user)):
    """与「文档上传」页同一条链路：保存原文 → 解析 → 切块 → 嵌入入库。

    去重挡在 add_document **之前**：指纹落库后再发现重复，就只能在
    「留垃圾记录」和「删数据」之间二选一。返回 409，正文说明重复。
    """
    from src.chunker import chunk_parsed
    from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, kb_upload_dir
    from src.contextualizer import build_full_document, contextualize_chunks
    from src.database import add_document, check_duplicate, compute_content_hash, get_kb, update_document_status
    from src.parser import parse_file
    from src.vector_store import add_chunks

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")
    require_kb_access(kb_id, user, "editor")

    name = file.filename or "unnamed"
    if not any(name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400,
                            detail=f"不支持的文件类型，允许: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    size = len(content)
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件超过 {MAX_FILE_SIZE_MB}MB 限制")

    # 原始文件必须留存：它是「重建索引」的唯一素材（改了分块/嵌入参数后要靠它重跑）
    dest = kb_upload_dir(kb_id) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_existed = dest.exists()  # 同名重传时 dest 覆盖的是**已有文档**的原文，重复时绝不能删
    dest.write_bytes(content)

    doc_id: str | None = None  # parse 在 add_document 之前就可能抛错，此时还没有记录可标 error
    try:
        parsed = parse_file(str(dest))
        if not parsed:
            doc_id = add_document(kb_id, name, size)
            update_document_status(doc_id, "empty")
            return {"doc_id": doc_id, "filename": name, "status": "empty",
                    "message": "没有可提取的文本内容"}

        chunks = chunk_parsed(parsed)
        if not chunks:
            doc_id = add_document(kb_id, name, size)
            update_document_status(doc_id, "empty")
            return {"doc_id": doc_id, "filename": name, "status": "empty",
                    "message": "内容太短，无法分块"}

        content_hash = compute_content_hash(chunks)
        if check_duplicate(kb_id, content_hash):
            if not file_existed:
                dest.unlink(missing_ok=True)  # 新文件名同内容：删掉孤儿原文，不占磁盘
            raise HTTPException(status_code=409, detail=f"内容与库中已有文件重复，已跳过入库: {name}")

        # Contextual Retrieval：入库前给 chunk 拼文档级上下文前缀（开关关闭时 no-op）。
        # 必须在指纹计算**之后**——LLM 输出不确定，加了前缀再算指纹，
        # 同一文件重传可能因描述措辞不同而绕过查重。
        chunks = contextualize_chunks(chunks, build_full_document(parsed))

        doc_id = add_document(kb_id, name, size, content_hash)
        add_chunks(kb_id, chunks)
        update_document_status(doc_id, "ready", len(chunks))
        return {
            "doc_id": doc_id, "filename": name, "status": "ready",
            "paragraphs": len(parsed), "chunks": len(chunks),
            "image_chunks": sum(1 for c in chunks if c.get("type") == "image"),
        }
    except HTTPException:
        raise
    except Exception as e:
        if doc_id is not None:
            update_document_status(doc_id, "error")
        raise HTTPException(status_code=500, detail=f"{name} 处理失败: {e}")


@router.post("/kbs/{kb_id}/reindex", summary="重建索引")
def reindex(kb_id: str, user: dict = Depends(get_current_user)):
    """用保留的原文按当前分块/嵌入参数重跑全库（src/reindex.py）。"""
    from src.database import get_kb
    from src.reindex import rebuild_kb

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")
    require_kb_access(kb_id, user, "editor")
    return rebuild_kb(kb_id)


@router.delete("/docs/{doc_id}", summary="删除文档")
def delete_doc(doc_id: str, user: dict = Depends(get_current_user)):
    """SQLite 记录 + Chroma chunk 双清，避免留下「列表里没了、检索还能搜到」的幽灵引用。

    权限顺序很重要：**先查 → 再判权限 → 最后才删**。若沿用「先 delete_document
    再用它的返回值拿 kb_id」，等发现没权限时数据已经没了，权限检查就成了摆设。
    """
    from src.database import delete_document, get_document
    from src.vector_store import delete_chunks_by_source

    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    require_kb_access(doc["kb_id"], user, "editor")
    delete_document(doc_id)
    deleted = delete_chunks_by_source(doc["kb_id"], doc["filename"])
    return {"doc_id": doc_id, "filename": doc["filename"], "deleted_chunks": deleted}


@router.get("/answer_log", summary="问答日志 + 缺口清单")
def answer_log_api(kb_id: str | None = None, limit: int = 50,
                   only_ungrounded: bool = False, only_rated: bool = False,
                   user: dict = Depends(get_current_user)):
    """问答日志，可按两类信号筛选（筛选在下层 SQL 完成，不拉全表再过滤）。

    · `only_ungrounded` —— 隐式信号：门控没过（库里撑不住这个问法）
    · `only_rated`      —— 显式信号：用户点过 👍/👎 的记录（含"答上了但没用"）

    权限：不传 kb_id 等于要看**所有库**的日志，这是跨库视图，只开放给 admin。
    普通用户必须显式指定自己有权限的 kb_id——默认拒绝比默认放行安全。
    """
    from src.answer_log import count_answers, gap_stats, list_answers

    if kb_id:
        require_kb_access(kb_id, user, "viewer")
    elif user.get("role") != "admin":
        raise HTTPException(status_code=400, detail="请指定 kb_id（跨库视图仅管理员可用）")

    return {
        "kb_id": kb_id,
        "stats": gap_stats(kb_id),
        "recent": list_answers(kb_id, limit=limit,
                               only_ungrounded=only_ungrounded, only_rated=only_rated),
        "total": count_answers(kb_id),
    }


@router.get("/eval/history", summary="RAGAS 评估历史存档（摘要列表）")
def eval_history(user: dict = Depends(get_current_user)):
    from src.evaluation_ragas import list_history

    return list_history()


@router.get("/eval/history/{filename}", summary="加载某一份评估存档的完整结果")
def eval_history_detail(filename: str, user: dict = Depends(get_current_user)):
    from src.evaluation_ragas import load_history

    data = load_history(filename)
    if data is None:
        raise HTTPException(status_code=404, detail=f"评估存档不存在: {filename}")
    return data
