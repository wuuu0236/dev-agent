"""
检索质量评测脚本 —— 批次 0：给 RAG 优化装仪表盘

做什么：
  用固定测试集（tests/golden_set.json）跑一遍检索，算出检索侧指标，
  作为后续所有 RAG 优化的基线。没有基线，任何"优化"都无法证明有效。

指标：
  Recall@K  期望来源被召回的比例（检索能找到多少）
  Hit@K     前 K 条里是否至少命中一个（用户能不能看到答案）
  MRR       第一个命中结果的排名倒数（好答案排得有多靠前）
  负样本    期望低分/无命中，用于将来验证"相似度阈值拒答"

用法：
  conda activate dev-agent
  python scripts/eval_retrieval.py                     # 默认知识库 + top_k=5
  python scripts/eval_retrieval.py --kb f99c2c78 --top-k 10
  python scripts/eval_retrieval.py --verbose           # 打印每条命中详情
  python scripts/eval_retrieval.py --compare <旧结果.json>   # 与历史对比

输出：
  控制台表格 + JSON 存档（data/eval_history/retrieval_eval_<时间>.json）

注意：
  本脚本只测"检索"，不调 LLM 生成答案（RAGAS 生成侧指标走评估面板）。
  检索是生成的上限——检索召不回，生成再好也没用，所以先看这个。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "golden_set.json"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "eval_history"


def load_golden(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_score(results: list) -> float:
    """取首位结果的相关性分数。

    优先读精排分数（rerank_score）——因为开了精排之后，结果**就是按它排序的**，
    再读 rrf_score 等于看一个跟顺序无关的数。精排关闭/降级时结果里没有该字段，
    自然回退到粗排的 rrf_score，行为与改动前一致。
    """
    if not results:
        return 0.0
    r = results[0]
    return r.get("rerank_score", r.get("rrf_score", 0.0))


def evaluate(kb_id: str, cases: list, top_k: int, verbose: bool,
             rerank: bool | None = None) -> dict:
    """跑检索并计算指标。

    rerank: True/False 显式指定精排开关（做 A/B），None 则跟随 config.RERANK_ENABLED。
    """
    from src.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(kb_id)
    details = []

    for case in cases:
        q = case["question"]
        expected = set(case.get("expected_sources") or [])
        is_negative = case.get("type") == "negative"

        results = retriever.search(q, top_k=top_k, rerank=rerank)
        got_sources = [r["source"] for r in results]

        # 命中的期望来源（按结果顺序）
        hits = [s for s in got_sources if s in expected]

        if is_negative:
            recall = None
            rr = None
            hit_at_1 = None
            score = top_score(results)
        else:
            recall = len(set(hits)) / len(expected) if expected else 0.0
            rr = 0.0
            for i, s in enumerate(got_sources, start=1):
                if s in expected:
                    rr = 1.0 / i
                    break
            hit_at_1 = 1.0 if (got_sources and got_sources[0] in expected) else 0.0
            score = top_score(results)

        details.append({
            "id": case["id"],
            "question": q,
            "type": case.get("type", "single-hop"),
            "expected": sorted(expected),
            "got": got_sources[:top_k],
            "recall": recall,
            "mrr": rr,
            "hit@1": hit_at_1,
            "top_score": score,
            "missed": sorted(expected - set(hits)) if not is_negative else [],
        })

        if verbose:
            flag = "OK " if (is_negative or recall == 1.0) else "MISS"
            print(f"  [{flag}] {case['id']} {q[:36]}")
            if not is_negative and recall < 1.0:
                print(f"        期望 {sorted(expected)} 实得 {got_sources[:3]}")

    # --- 汇总（只统计正样本）---
    pos = [d for d in details if d["recall"] is not None]
    neg = [d for d in details if d["recall"] is None]
    n = len(pos) or 1

    summary = {
        "cases_total": len(details),
        "cases_positive": len(pos),
        "cases_negative": len(neg),
        f"Recall@{top_k}": round(sum(d["recall"] for d in pos) / n, 4),
        f"Hit@{top_k}": round(sum(1 for d in pos if d["recall"] > 0) / n, 4),
        "Hit@1": round(sum(d["hit@1"] for d in pos) / n, 4),
        "MRR": round(sum(d["mrr"] for d in pos) / n, 4),
        "negative_top_score_max": round(max([d["top_score"] for d in neg], default=0.0), 4),
    }

    return {"summary": summary, "details": details}


def print_table(result: dict, top_k: int, kb_id: str, kb_name: str,
                rerank: bool | None = None):
    s = result["summary"]
    print()
    print("=" * 56)
    print(f"检索质量基线  |  知识库: {kb_name} ({kb_id})")
    print("=" * 56)
    print(f"  用例数        {s['cases_total']}（正样本 {s['cases_positive']} / 负样本 {s['cases_negative']}）")
    print(f"  Recall@{top_k}     {s[f'Recall@{top_k}']:.3f}   期望来源被召回的比例")
    print(f"  Hit@{top_k}        {s[f'Hit@{top_k}']:.3f}   前 {top_k} 条内至少命中一个")
    print(f"  Hit@1         {s['Hit@1']:.3f}   第一条就命中")
    print(f"  MRR           {s['MRR']:.3f}   命中结果排名的倒数均值")
    if rerank:
        hint = "精排分数 0~1 有绝对意义，可直接做阈值拒答"
    else:
        hint = "RRF 分数 1/(60+rank) 无绝对意义，仅作相对参考"
    print(f"  负样本最高分  {s['negative_top_score_max']:.4f}  （{hint}）")
    print("=" * 56)

    missed = [d for d in result["details"] if d["missed"]]
    if missed:
        print(f"\n未完全命中 {len(missed)} 条（按归因排查：解析 / 分块 / 检索）：")
        for d in missed[:10]:
            print(f"  · {d['id']} {d['question'][:32]}")
            print(f"    期望 {d['expected']} → 实得 {d['got'][:3]}")
    print()


def main():
    ap = argparse.ArgumentParser(description="RAG 检索质量评测（批次 0 基线）")
    ap.add_argument("--kb", default=None, help="知识库 ID，默认读 golden_set.json 的 meta.kb_id")
    ap.add_argument("--top-k", type=int, default=5, help="检索返回条数（默认 5）")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="测试集路径")
    ap.add_argument("--verbose", action="store_true", help="打印每条用例命中详情")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--compare", default=None, help="与历史结果 JSON 对比")
    ap.add_argument("--no-archive", action="store_true", help="不写存档文件")
    # 精排开关：默认跟随 config.RERANK_ENABLED。做 A/B 时显式指定，避免"改了配置
    # 默认值就再也测不出对照"的问题。
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                     help="强制开启精排")
    grp.add_argument("--no-rerank", dest="rerank", action="store_false",
                     help="强制关闭精排（跑无精排的对照基线）")
    args = ap.parse_args()

    golden = load_golden(Path(args.golden))
    kb_id = args.kb or golden["meta"]["kb_id"]
    kb_name = golden["meta"].get("kb_name", "")
    cases = golden["cases"]
    if args.limit:
        cases = cases[: args.limit]

    # 实际生效的检索配置（None 时读配置），存档里要记下来，否则两次结果没法归因
    from src.config import BM25_WEIGHT, RERANK_ENABLED, VECTOR_WEIGHT
    rerank_effective = RERANK_ENABLED if args.rerank is None else args.rerank
    routes = "向量" + (f" + BM25(权重 {BM25_WEIGHT:g})" if BM25_WEIGHT else "（BM25 关）")

    print(f"加载测试集: {args.golden}")
    print(f"用例数 {len(cases)} | 知识库 {kb_id} | top_k={args.top_k} | "
          f"精排={'开' if rerank_effective else '关'} | 召回: {routes}")
    print(f"  提示：切 BM25 做 A/B 用环境变量，例如 BM25_WEIGHT=0 python {Path(__file__).name} --no-archive")

    result = evaluate(kb_id, cases, args.top_k, args.verbose, rerank=args.rerank)
    print_table(result, args.top_k, kb_id, kb_name, rerank_effective)

    # --- 与历史对比 ---
    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_s = old["summary"]
        new_s = result["summary"]
        print("与历史对比：")
        for key in [f"Recall@{args.top_k}", "Hit@1", "MRR"]:
            if key in old_s and key in new_s:
                delta = new_s[key] - old_s[key]
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                print(f"  {key:<12} {old_s[key]:.3f} → {new_s[key]:.3f}  {arrow} {delta:+.3f}")
        print()

    # --- 存档 ---
    if not args.no_archive:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ARCHIVE_DIR / f"retrieval_eval_{ts}.json"
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "kb_id": kb_id,
            "kb_name": kb_name,
            "top_k": args.top_k,
            "rerank": rerank_effective,
            "bm25_weight": BM25_WEIGHT,
            "vector_weight": VECTOR_WEIGHT,
            "golden_set": str(args.golden),
            **result,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"存档: {out.relative_to(PROJECT_ROOT)}")
        print(f"下次对比: python scripts/eval_retrieval.py --top-k {args.top_k} --compare \"{out}\"")


if __name__ == "__main__":
    main()
