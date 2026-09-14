"""
重建索引——用保留的原始文件，把知识库按**当前**的解析 / 清洗 / 分块 / 嵌入配置重跑一遍

为什么需要：
  索引一旦写进向量库，就被"冻"在当时那套参数上了。之后你调了分块长度、换了清洗
  规则、改了嵌入模型，**老文档不会自动跟着变**——库里还是旧规则切出来的块。
  这时只有两条路：重新上传（要求你手上还有原始文件），或者重建索引（一次计算）。
  `vector_store.check_embedding_dim()` 的提示语一直是"请重新上传文档重建此库"，
  就是缺了后面这条路。

为什么必须先保留原文：
  **没有原文就没法重建。** 改分块参数要重新切分，而切分只能从原文来。
  此前实现把非图片文件在上传后直接删掉了（`pages/2` 的 finally），等于永久放弃了
  这项能力——所以 2026-09-14 的上传改造与"保留原文"是同一件事的两面。

三种结果必须分清楚（每次重建都如实报告）：
  · ok       成功重建
  · missing  原始文件不在 → **无法重建**，旧索引保持原样（不会被清掉）
  · failed   解析 / 切分 / 入库出错 → 旧索引同样保持原样，可以重试

安全保证：
  · 逐文档替换，不做"整库清空再写"——一个文件出问题不会连累其他文档
  · 写入走 `add_chunks()`（先算完向量再删旧块），任何一次 embedding 失败都不会
    让已有内容受损
"""
from collections.abc import Callable

from src.config import kb_upload_dir


def rebuildable_docs(kb_id: str) -> tuple[list[str], list[str]]:
    """返回 (有原文、可重建的文档名, 缺原文、无法重建的文档名)。

    给界面提前提示用：**用户应该在做之前就知道有几个文档重建不了**，
    而不是等跑完才被告知。
    """
    from src.database import list_documents

    upload_dir = kb_upload_dir(kb_id)
    ready, missing = [], []
    for doc in list_documents(kb_id):
        (ready if (upload_dir / doc["filename"]).exists() else missing).append(doc["filename"])
    return ready, missing


def rebuild_kb(kb_id: str, progress: Callable[[str], None] | None = None) -> dict:
    """按当前配置重建整个知识库的索引。

    progress: 可选回调，界面用它显示正在处理哪个文件。
    返回 {total, ok, chunks, missing, failed}。
    """
    from src.database import list_documents, update_document_status
    from src.parser import parse_file
    from src.chunker import chunk_parsed
    from src.vector_store import add_chunks

    docs = list_documents(kb_id)
    upload_dir = kb_upload_dir(kb_id)

    ok, total_chunks = 0, 0
    missing: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, doc in enumerate(docs, 1):
        name = doc["filename"]
        if progress:
            progress(f"[{i}/{len(docs)}] {name}")

        path = upload_dir / name
        if not path.exists():
            # 原文不在（改造前入库的老数据）：只记录，不动它的旧索引
            missing.append(name)
            continue

        try:
            parsed = parse_file(str(path))          # 内含清洗
            chunks = chunk_parsed(parsed) if parsed else []
            if not chunks:
                update_document_status(doc["id"], "empty")
                failed.append((name, "没有可用的文本内容"))
                continue

            add_chunks(kb_id, chunks)               # replace_source 默认 True：同源旧块被替换
            update_document_status(doc["id"], "ready", len(chunks))
            ok += 1
            total_chunks += len(chunks)
        except Exception as e:
            # 失败 = "什么都没发生"，旧索引保持原样，而不是"内容没了"
            failed.append((name, f"{type(e).__name__}: {e}"))

    return {
        "total": len(docs),
        "ok": ok,
        "chunks": total_chunks,
        "missing": missing,
        "failed": failed,
    }
