"""导入期不得要求凭据。

守的是什么：
  `import src.vector_store` 这类动作**不应该需要任何 API key**。凭据缺失只在
  「真的去调 API」时才该报错，而不是在导入时。

为什么值得单开一个文件：
  2026-09-14 的 CI 事故就是这条被破坏的结果 —— `src/embeddings.py` 在模块级写了
  `_client = OpenAI(api_key=EMBEDDING_API_KEY, ...)`，而 `OpenAI(api_key="")` 会当场抛
  `OpenAIError: Missing credentials`（openai 2.x / 3.x 行为一致，已实测）。
  CI 里没有任何密钥，于是三个测试文件**只是 import 就**在收集阶段集体 ImportError，
  pytest 退出码 2，整个 test job 红了一个月。
  本地有 .env 所以永远复现不出来 —— 这正是它必须由测试而不是靠人守的原因。

为什么要在子进程里跑：
  本机 .env 里有真 key，同进程里怎么断言都会通过。子进程里把 key 显式置空，
  才等价于 CI。

为什么给 langfuse 塞替身：
  本文件要隔离的变量**只有一个：有没有凭据**。缺 langfuse / langchain_openai 属于另一类
  问题（langfuse 的降级由 test_smoke_imports 守）。不塞替身的话，本机缺 langfuse
  会让这个文件整片红，等于把两个不相干的变量搅在一起。
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 这些模块必须能在「没有任何凭据」的环境里被导入。
# 取舍：只列**不依赖可选 SDK**的模块 —— 需要 langchain_openai / mcp 的那几个
# （src.evaluation_ragas、src.agent.dev_agent_langgraph、src.mcp_server）由别的测试守，
# 否则本文件会在本机因缺 langchain_openai 而红，掩盖真正的主题。
MODULES = [
    "src.embeddings",
    "src.vector_store",
    "src.reindex",
    "src.parser",
    "src.chunker",
    "src.cleaner",
    "src.hybrid_retriever",
    "src.citations",
    "src.database",
    "src.answer_gate",
    "src.answer_log",
    "src.query_cache",
    "src.reranker",
    "src.query_rewrite",
    "src.evaluation",          # 经 src.rag_qa 依赖 langfuse → 靠下面替身兜住
    "src.tools.safety",
    "src.tools.file_tools",
]

_CHILD = """
import importlib, sys, types

MODULES = __MODULES__

# 只补「可选 SDK」的替身，让本测试专注在凭据这一个变量上。
try:
    import langfuse  # noqa: F401
except Exception:
    sys.modules.setdefault("langfuse", types.ModuleType("langfuse"))


def _observe(*a, **kw):
    return a[0] if (a and callable(a[0])) else (lambda fn: fn)


_dec = types.ModuleType("langfuse.decorators")
_dec.observe = _observe
sys.modules["langfuse.decorators"] = _dec

bad = []
for name in MODULES:
    try:
        importlib.import_module(name)
    except Exception as e:
        bad.append("  %s -> %s: %s" % (name, type(e).__name__, e))

if bad:
    print("以下模块在没有凭据时无法导入：", file=sys.stderr)
    print(chr(10).join(bad), file=sys.stderr)
    raise SystemExit(1)

# 反面：**用**的时候必须真的失败。否则「修好导入」会退化成「静默拿个假客户端」，
# 那比导入报错更危险 —— 线上会以为在调模型，其实没有。
import src.embeddings as _emb
try:
    _emb._get_client()
except Exception:
    pass
else:
    raise SystemExit("凭据为空却拿到了客户端：缺失被静默吞掉了")

print("ok")
"""


def _run_child() -> subprocess.CompletedProcess:
    """在「无凭据」的子进程里跑导入检查。

    ⚠️ 显式把 key 置空即可，不需要动 .env：`load_dotenv()` 默认不覆盖已存在的环境变量，
    所以已置空的变量不会被 .env 里的真值补回来。
    """
    import os

    env = {
        **os.environ,
        "EMBEDDING_API_KEY": "",
        "DEEPSEEK_API_KEY": "",     # EMBEDDING_API_KEY 会回退到它
        "RERANK_API_KEY": "",
        "LLM_API_KEY": "",
    }
    return subprocess.run(
        [sys.executable, "-c", _CHILD.replace("__MODULES__", repr(MODULES))],
        cwd=REPO, env=env, capture_output=True, text=True,
    )


def test_modules_import_without_any_credentials():
    r = _run_child()
    assert r.returncode == 0, (
        "无凭据环境下导入失败 —— 说明有模块在**导入期**就要求 API key。\n"
        "修法：把客户端构造改成惰性（首次调用时才建），参考 "
        "src/embeddings.py::_get_client。\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )


# 不做「每个模块单独起一个子进程」的并集：上面那条已经逐个 import 并逐条列名，
# 隔离版只会把测试时长翻倍（17 次解释器启动），换不来新的信息。
