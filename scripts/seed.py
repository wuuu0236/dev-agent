"""
seed.py — 预置演示知识库与种子文档

为什么需要：
  data/ 目录不进 git（.gitignore），所以新部署 / Streamlit Cloud 冷启动时，
  知识库是空的——评审打开链接看到"请先创建知识库"就卡住了。
  本脚本幂等地补一份演示知识库，让项目开箱即用。

做法（幂等，可重复跑）：
  1. 确保存在名为「DataLens 演示」的知识库。已存在则复用——绝不重建 collection
     （create_collection 是"先删再建"，对已有库调用会清空用户数据）。
  2. 把 scripts/seed_data/ 下的文档走完整上传链路导入（parse → chunk → add_chunks）。
  3. 按文件名（seed_ 前缀）判断是否已导入，已导入直接跳过。

两种用法：
  python scripts/seed.py              # 命令行手动跑
  app.py 启动时调用 ensure_seed_data()  # Streamlit 冷启动自动补种子
"""
import sys
from pathlib import Path

# 让 `from src.* import ...` 在任意目录下都能跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import (init_db, list_kbs, create_kb, add_document,
                          update_document_status, list_documents)
from src.vector_store import create_collection, add_chunks
from src.parser import parse_file
from src.chunker import chunk_parsed

SEED_KB_NAME = "DataLens 演示"
SEED_DIR = Path(__file__).resolve().parent / "seed_data"


def _ensure_kb() -> str:
    """确保存在种子知识库，返回 kb_id。

    只在【新建】知识库时创建 collection。已有的种子库直接复用，
    不调用 create_collection——那会先删掉再重建，清空已有数据。
    """
    for kb in list_kbs():
        if kb["name"] == SEED_KB_NAME:
            return kb["id"]
    kb = create_kb(SEED_KB_NAME, "预置演示知识库（scripts/seed.py 自动导入）")
    create_collection(kb["id"])
    return kb["id"]


def _already_imported(kb_id: str) -> bool:
    """种子文档是否已导入：库里存在 seed_ 前缀且状态 ready 的文档即视为已导入。"""
    docs = list_documents(kb_id)
    return any(d["filename"].startswith("seed_") and d["status"] == "ready"
               for d in docs)


def ensure_seed_data(verbose: bool = True) -> str:
    """幂等导入种子数据。可重复调用；返回一句状态描述。"""
    init_db()
    kb_id = _ensure_kb()

    if _already_imported(kb_id):
        msg = f"[seed] 种子数据已存在，跳过（知识库：{SEED_KB_NAME}）"
        if verbose:
            print(msg)
        return msg

    imported = []
    for md_file in sorted(SEED_DIR.glob("*.md")):
        filename = f"seed_{md_file.name}"  # 数据库里带 seed_ 前缀便于识别/幂等
        doc_id = add_document(kb_id, filename, md_file.stat().st_size)
        try:
            parsed = parse_file(str(md_file))   # 复用真实解析链路
            if not parsed:
                update_document_status(doc_id, "empty")
                continue
            chunks = chunk_parsed(parsed)
            if not chunks:
                update_document_status(doc_id, "empty")
                continue
            add_chunks(kb_id, chunks)           # embedding + 写入向量库
            update_document_status(doc_id, "ready", len(chunks))
            imported.append(f"{filename} → {len(chunks)} chunks")
        except Exception as e:
            # 单个文档失败不影响其他文档；状态标记 error 便于排查
            update_document_status(doc_id, "error")
            if verbose:
                print(f"[seed] 导入失败 {filename}: {type(e).__name__}: {e}")

    if verbose:
        for line in imported:
            print(f"[seed] 已导入: {line}")
    msg = f"[seed] 导入完成：{len(imported)} 个文档 → 知识库「{SEED_KB_NAME}」"
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    ensure_seed_data()
