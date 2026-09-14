"""
Web API —— 给 React 前端（frontend/）提供 REST 端点

设计原则：
  · 这一层**只做协议转换**（HTTP ↔ 现有模块调用），不含任何业务逻辑。
    检索、门控、解析、入库全部复用 Streamlit 已有的模块，保证两条入口行为一致。
  · 函数内部 import：任何模块在无 API key 环境下也必须能 import（CI 守卫），
    而本文件只被 API 服务加载，运行时 import 不影响那条守卫。

端点一览：
  GET    /api/health                  服务可用性与后端连通状态
  GET    /api/kbs                     知识库列表（含文档数 / chunk 数）
  GET    /api/kbs/{kb_id}/docs        某知识库的文档列表
  POST   /api/ask                     RAG 问答（改写 → 检索 → 精排 → 门控 → 生成 → 引用）
  POST   /api/ask/stream              同上，但答案用 SSE 逐块推送（前端打字机效果）
  POST   /api/kbs/{kb_id}/upload      上传并入库（解析 → 切块 → 嵌入）
  POST   /api/kbs/{kb_id}/reindex     按当前参数重建索引
  DELETE /api/docs/{doc_id}           删除文档（SQLite + Chroma 双清）
  GET    /api/answer_log              问答日志 + 缺口分类
  GET    /api/eval/history            RAGAS 评估历史存档
"""

import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


@router.get("/kbs", summary="知识库列表")
def get_kbs():
    from src.database import get_kb_stats, list_kbs

    result = []
    for kb in list_kbs():
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
def get_docs(kb_id: str):
    from src.database import get_kb, get_kb_stats, list_documents

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")
    return {"kb_id": kb_id, "stats": get_kb_stats(kb_id), "docs": list_documents(kb_id)}


@router.post("/ask", response_model=AskResponse, summary="RAG 问答")
def ask(request: AskRequest):
    """走与 Streamlit 完全相同的 rag_query 链路，返回答案 + 引用 + 门控信息。"""
    from src.answer_gate import top_score
    from src.answer_log import get_answer
    from src.database import get_kb
    from src.rag_qa import rag_query

    if not get_kb(request.kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {request.kb_id}")

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
def ask_stream(request: AskRequest):
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


@router.post("/feedback", summary="给某条回答打反馈（👍/👎）")
def feedback(request: FeedbackRequest):
    """把人工反馈写回 answer_log.rating。

    rating 只允许 None / 'up' / 'down'，非法值由 answer_log.set_rating 抛错后转 400。
    带反馈的记录不会被容量淘汰清理，因此这些标注是稳定的信号源。
    """
    from src.answer_log import set_rating

    try:
        hit = set_rating(request.log_id, request.rating)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not hit:
        raise HTTPException(status_code=404, detail=f"问答记录不存在: {request.log_id}")
    return {"ok": True, "log_id": request.log_id, "rating": request.rating}


@router.post("/kbs/{kb_id}/upload", summary="上传文档并入库")
async def upload(kb_id: str, file: UploadFile = File(...)):
    """与「文档上传」页同一条链路：保存原文 → 解析 → 切块 → 嵌入入库。"""
    from src.chunker import chunk_parsed
    from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, kb_upload_dir
    from src.database import add_document, get_kb, update_document_status
    from src.parser import parse_file
    from src.vector_store import add_chunks

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")

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
    dest.write_bytes(content)

    doc_id = add_document(kb_id, name, size)
    try:
        parsed = parse_file(str(dest))
        if not parsed:
            update_document_status(doc_id, "empty")
            return {"doc_id": doc_id, "filename": name, "status": "empty",
                    "message": "没有可提取的文本内容"}

        chunks = chunk_parsed(parsed)
        if not chunks:
            update_document_status(doc_id, "empty")
            return {"doc_id": doc_id, "filename": name, "status": "empty",
                    "message": "内容太短，无法分块"}

        add_chunks(kb_id, chunks)
        update_document_status(doc_id, "ready", len(chunks))
        return {
            "doc_id": doc_id, "filename": name, "status": "ready",
            "paragraphs": len(parsed), "chunks": len(chunks),
            "image_chunks": sum(1 for c in chunks if c.get("type") == "image"),
        }
    except Exception as e:
        update_document_status(doc_id, "error")
        raise HTTPException(status_code=500, detail=f"{name} 处理失败: {e}")


@router.post("/kbs/{kb_id}/reindex", summary="重建索引")
def reindex(kb_id: str):
    """用保留的原文按当前分块/嵌入参数重跑全库（src/reindex.py）。"""
    from src.database import get_kb
    from src.reindex import rebuild_kb

    if not get_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库不存在: {kb_id}")
    return rebuild_kb(kb_id)


@router.delete("/docs/{doc_id}", summary="删除文档")
def delete_doc(doc_id: str):
    """SQLite 记录 + Chroma chunk 双清，避免留下「列表里没了、检索还能搜到」的幽灵引用。"""
    from src.database import delete_document
    from src.vector_store import delete_chunks_by_source

    doc = delete_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    deleted = delete_chunks_by_source(doc["kb_id"], doc["filename"])
    return {"doc_id": doc_id, "filename": doc["filename"], "deleted_chunks": deleted}


@router.get("/answer_log", summary="问答日志 + 缺口清单")
def answer_log_api(kb_id: str | None = None, limit: int = 50):
    from src.answer_log import gap_stats, list_answers

    return {
        "kb_id": kb_id,
        "stats": gap_stats(kb_id),
        "recent": list_answers(kb_id, limit=limit),
    }


@router.get("/eval/history", summary="RAGAS 评估历史存档（摘要列表）")
def eval_history():
    from src.evaluation_ragas import list_history

    return list_history()


@router.get("/eval/history/{filename}", summary="加载某一份评估存档的完整结果")
def eval_history_detail(filename: str):
    from src.evaluation_ragas import load_history

    data = load_history(filename)
    if data is None:
        raise HTTPException(status_code=404, detail=f"评估存档不存在: {filename}")
    return data
