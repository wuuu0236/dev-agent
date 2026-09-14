"""CI 环境自检：逐个导入核心依赖，把「哪个包没装上 / 装上了但导入就崩」直接打出来。

为什么需要单独一步：
  缺一个包时 pytest 只给一句 `Interrupted: N error during collection` + 退出码 2，
  再加上 GitHub 的 `Process completed with exit code 2.`，完全看不出是哪个模块。
  2026-09-14 排查时，这个现象让人在「缺依赖 / 版本不兼容 / 真的测试失败」之间反复猜，
  而 job 日志走 API 要鉴权、拿不到。

设计取舍：
  · **不 fail-fast**：逐个 import 并各自 try —— 一次跑完就能看到全部问题，
    而不是修一个再等 3 分钟看下一个。全部检查完再决定退出码。
  · **同时打印版本**：`ImportError` 和「版本组合不对」是两类问题，
    只报能否导入会漏掉后者（例如约束写成 `<1.0` 而代码按 1.x 写的）。
  · 只做检查、不修改环境，本地也能跑：`python scripts/check_ci_deps.py`
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import sys

# 模块名 → 发行包名（两者不同时必须写明，否则报版本时会显示"未安装"）
CORE = {
    "chromadb": "chromadb",
    "docx": "python-docx",
    "fitz": "PyMuPDF",
    "langgraph": "langgraph",
    "langchain_openai": "langchain-openai",
    "langchain_core": "langchain-core",
    "langfuse": "langfuse",
    "numpy": "numpy",
    "rank_bm25": "rank-bm25",
    "jieba": "jieba",
    "dotenv": "python-dotenv",
    "httpx": "httpx",
    "pytest": "pytest",
}


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "未安装"


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.platform})")
    print(f"解释器 {sys.executable}\n")

    failed: list[str] = []
    errs: dict[str, str] = {}
    print(f"{'模块':<18} {'发行包版本':<16} 结果")
    print("-" * 66)
    for mod, dist in CORE.items():
        try:
            importlib.import_module(mod)
            print(f"{mod:<18} {_version(dist):<16} OK")
        except Exception as e:  # 不只是 ImportError：装上了也可能导入期崩
            msg = f"{type(e).__name__}: {e}"
            print(f"{mod:<18} {_version(dist):<16} FAIL  {msg}")
            failed.append(mod)
            errs[mod] = msg

    print()
    if failed:
        print("❌ 以下依赖在本环境不可用：" + ", ".join(failed))
        print("   若是 ModuleNotFoundError → requirements-dev.txt 漏声明；")
        print("   若是别的异常 → 装上了但版本组合不兼容，看上面的版本号。")
        # 发一条 GitHub annotation：job 日志走 API 要鉴权（403），
        # 而 check-run 的 annotations 是**匿名可读**的 —— 这是没装 gh、也无法登录时
        # 唯一能看到失败原文的通道。%0A 是工作流命令里的换行转义。
        detail = "%0A".join(f"{m} ({_version(CORE[m])}): {errs[m]}" for m in failed)
        print(f"::error title=CI 依赖自检失败::{detail}")
        return 1
    print("✅ 核心依赖全部可导入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
