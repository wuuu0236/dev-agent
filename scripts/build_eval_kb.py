"""
构建「RAG 评测语料库」—— 让测试集这把尺子重新变得灵敏

背景（实测发现的问题）：
  原知识库只有 6 个文档 / 18 个 chunk，top_k=5 相当于捞了全库 28%，
  Recall@5 恒等于 1.0——任何优化都测不出差别（改前改后指标一模一样）。
  检索评测必须有"大海捞针"的难度才有区分度。

做什么：
  把项目里所有 Markdown 文档（knowledge / notes / docs / 根目录 README 等）
  全部导入一个新建的知识库，把语料撑到几百 chunk。
  原有的 27 条测试问题（针对 knowledge/ 那 5 篇）依然有效，
  但检索难度大幅提升，优化效果就能被测出来了。

用法：
  python scripts/build_eval_kb.py                # 建库并导入，打印 kb_id
  python scripts/build_eval_kb.py --dry-run      # 只看会导入哪些文件

输出：
  控制台打印知识库 ID，并提示如何接上评测：
  python scripts/eval_retrieval.py --kb <kb_id>
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 参与建库的目录（相对项目根）
SOURCE_DIRS = ["knowledge", "notes", "docs", "scripts/seed_data"]
# 参与建库的根目录单文件
SOURCE_FILES = ["README.md", "README_CN.md", "CLAUDE.md", "USE-CASES.md", "CONTRIBUTING.md"]
# 支持的扩展名（走 parse_txt）
EXTS = {".md", ".txt", ".csv"}

KB_NAME = "RAG 评测语料库"


def collect_files() -> list[Path]:
    """收集所有待导入的文档。"""
    files = []
    for d in SOURCE_DIRS:
        base = PROJECT_ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                files.append(p)
    for f in SOURCE_FILES:
        p = PROJECT_ROOT / f
        if p.is_file():
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="构建 RAG 评测语料库")
    ap.add_argument("--dry-run", action="store_true", help="只列出将要导入的文件，不执行")
    ap.add_argument("--kb-name", default=KB_NAME, help="知识库名称")
    args = ap.parse_args()

    files = collect_files()
    if not files:
        print("没有找到可导入的文档，检查 SOURCE_DIRS 配置。")
        return

    total_bytes = sum(p.stat().st_size for p in files)
    print(f"待导入 {len(files)} 个文档，共 {total_bytes / 1024:.1f} KB")
    if args.dry_run:
        for p in files:
            print(f"  · {p.relative_to(PROJECT_ROOT)}")
        return

    from src.database import init_db, create_kb, add_document, update_document_status
    from src.parser import parse_file
    from src.chunker import chunk_parsed
    from src.vector_store import create_collection, add_chunks, collection_count

    init_db()
    kb = create_kb(args.kb_name, "用于检索质量评测：项目内全部 Markdown 文档，撑大语料让测试集有区分度")
    kb_id = kb["id"]
    create_collection(kb_id)
    print(f"已创建知识库 {args.kb_name} (id={kb_id})")

    ok, empty, fail, total_chunks = 0, 0, 0, 0
    for p in files:
        rel = p.relative_to(PROJECT_ROOT)
        try:
            parsed = parse_file(str(p))
            if not parsed:
                print(f"  [空] {rel}")
                empty += 1
                continue
            chunks = chunk_parsed(parsed)
            if not chunks:
                print(f"  [空] {rel}")
                empty += 1
                continue
            doc_id = add_document(kb_id, p.name, p.stat().st_size)
            add_chunks(kb_id, chunks)
            update_document_status(doc_id, "ready", len(chunks))
            total_chunks += len(chunks)
            ok += 1
            print(f"  [OK] {rel} → {len(chunks)} chunks")
        except Exception as e:
            fail += 1
            print(f"  [失败] {rel}: {type(e).__name__}: {e}")

    print()
    print("=" * 52)
    print(f"导入完成：成功 {ok} / 空 {empty} / 失败 {fail}")
    print(f"知识库 chunk 总数：{collection_count(kb_id)}")
    print(f"知识库 ID：{kb_id}")
    print("=" * 52)
    print("下一步：把该 ID 填进 tests/golden_set.json 的 meta.kb_id，然后")
    print(f"  python scripts/eval_retrieval.py --kb {kb_id}")


if __name__ == "__main__":
    main()
