# -*- coding: utf-8 -*-
"""
ingest_missing.py — 把 knowledge/ 下缺失的文档补进「DataLens 演示」知识库

背景：
  评估实验（run_eval_experiment.py）发现 MCP×3、版本迭代、技术栈、Docker 这几题
  检索返回"知识库中未找到相关信息"，原因是 demo 知识库（scripts/seed.py 从
  scripts/seed_data/rag_agent_guide.md 单文件灌入）未包含 MCP / 部署 相关内容。

  而 knowledge/04-mcp-protocol.md、05-dev-agent-deployment.md 两篇源文档是齐全的，
  本脚本把它们走真实链路（parse → chunk → embedding → add_chunks）导入现有 KB，
  补齐覆盖缺口，让评估分数真实反映系统能力。

幂等：同名文档已 ready 则跳过，可重复运行。
不删除任何已有文档，不影响其它已导入内容。

运行（在 dev-agent 目录下，需 .env 配好 EMBEDDING_API_KEY 且能联网）：
  python ingest_missing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database import list_kbs, list_documents, add_document, update_document_status
from src.parser import parse_file
from src.chunker import chunk_parsed
from src.vector_store import add_chunks

KB_NAME = "DataLens 演示"
SRC_DIR = Path(__file__).resolve().parent / "knowledge"
FILES = ["04-mcp-protocol.md", "05-dev-agent-deployment.md"]


def main():
    kb = next((k for k in list_kbs() if k["name"] == KB_NAME), None)
    if kb is None:
        print(f"[错误] 找不到知识库「{KB_NAME}」，请先跑一次 scripts/seed.py")
        return
    kb_id = kb["id"]
    existing = {d["filename"] for d in list_documents(kb_id) if d.get("status") == "ready"}

    for fn in FILES:
        if fn in existing:
            print(f"[跳过] {fn} 已存在（ready）")
            continue
        p = SRC_DIR / fn
        if not p.exists():
            print(f"[跳过] 源文件不存在: {p}")
            continue
        doc_id = add_document(kb_id, fn, p.stat().st_size)
        try:
            parsed = parse_file(str(p))
            if not parsed:
                update_document_status(doc_id, "empty")
                print(f"[空] {fn} 解析为空，跳过")
                continue
            chunks = chunk_parsed(parsed)
            if not chunks:
                update_document_status(doc_id, "empty")
                print(f"[空] {fn} 切块为空，跳过")
                continue
            add_chunks(kb_id, chunks)  # 内部做 embedding 并写入向量库
            update_document_status(doc_id, "ready", len(chunks))
            print(f"[ok] 已导入 {fn} → {len(chunks)} chunks")
        except Exception as e:
            update_document_status(doc_id, "error")
            print(f"[失败] {fn}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
