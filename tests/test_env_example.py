"""`.env.example` 与 `src/config.py` 的一致性守卫。

为什么需要：`.env.example` 是别人（含面试官复现环境时）唯一照抄的配置模板，
而它**已经漏过必填项**——`EMBEDDING_API_KEY` 没写。后果是照着配的人 embedding
必然 401：`config.py` 里它的回退值是 `LLM_API_KEY`（DeepSeek 的 key），
而 base_url 指向硅基流动，**两家 key 不通用**；更糟的是报错信息不会指向缺的
这一行，排查很绕。

这类"没人会为它写测试"的文件，一旦漏项就会长期漏着。本文件把
「config 需要的」与「模板给出的」绑起来。

只读文本 + 纯字符串比较，不导入重型依赖 → CI 可跑。
"""
import re
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
CONFIG_SRC = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
EXAMPLE = dotenv_values(str(ROOT / ".env.example"))

# 允许在模板里显式赋空值的键：语义上"空"就是一种有效取值。
# OLLAMA_VISION_MODEL 留空 = 不启用本地视觉模型（代码里按 falsy 判断）。
_EMPTY_ALLOWED = {"OLLAMA_VISION_MODEL"}

# 默认开启的功能开关。必须写进模板——否则使用者不知道怎么关掉它，
# 而这些开关都会改变行为（甚至影响成本与延迟）。
DEFAULT_ON_SWITCHES = [
    "RERANK_ENABLED",          # 精排（每次检索多一次 API 调用）
    "RETRIEVAL_MIN_SCORE",     # 质量门控（阈值拒答）
    "QUERY_REWRITE_ENABLED",   # 多轮追问消解
    "BM25_WEIGHT",             # 混合检索的一路
    "VECTOR_WEIGHT",
    "ENABLE_OCR",              # 本地 OCR
    "QUERY_CACHE_ENABLED",     # 语义缓存
]


def _env_vars_with_empty_default() -> list[str]:
    """config.py 里 `os.getenv("X", "")` 的 X —— 空默认值即"必须自己填"。"""
    return sorted(set(re.findall(
        r'os\.getenv\(\s*"([A-Z0-9_]+)"\s*,\s*""\s*\)', CONFIG_SRC
    )))


def _env_vars_falling_back_to_another_var() -> dict[str, str]:
    """config.py 里 `os.getenv("X", OTHER)` 的 {X: OTHER}。

    这类回退最危险：键缺失时不报错，而是**悄悄用了另一个服务的凭据**。
    """
    pairs = re.findall(
        r'os\.getenv\(\s*"([A-Z0-9_]+)"\s*,\s*([A-Z_][A-Z0-9_]*)\s*\)', CONFIG_SRC
    )
    return {name: fallback for name, fallback in pairs}


def _unsafe_key_fallbacks() -> dict[str, str]:
    """挑出「凭据借了别家、端点却是自家」的回退。

    判据不是"有没有回退"，而是**凭据与端点是否同源**：
      · RERANK_API_KEY ← EMBEDDING_API_KEY **且** RERANK_API_BASE ← EMBEDDING_API_BASE
        → 两者同源，服务一致，安全。
      · EMBEDDING_API_KEY ← LLM_API_KEY（DeepSeek），而 EMBEDDING_API_BASE 写死
        硅基流动 → **凭据与端点分属两家，必然 401**，且不填不会报错。
    """
    fallbacks = _env_vars_falling_back_to_another_var()
    unsafe: dict[str, str] = {}
    for name, source in fallbacks.items():
        if not name.endswith("_API_KEY"):
            continue
        base_var = name[: -len("_API_KEY")] + "_API_BASE"
        inherited_base = source[: -len("_API_KEY")] + "_API_BASE"
        if fallbacks.get(base_var) == inherited_base:
            continue  # 端点跟着一起继承 → 同源 → 安全
        unsafe[name] = source
    return unsafe


class TestRequiredKeysDocumented:
    def test_empty_default_keys_are_documented(self):
        """空默认值 = 不填就跑不起来 → 模板必须给出。"""
        missing = [k for k in _env_vars_with_empty_default() if k not in EXAMPLE]
        assert not missing, f"这些必填键在 .env.example 里没有：{missing}"

    def test_cross_service_fallback_keys_are_documented(self):
        """凭据与端点分属两家的回退，必须显式写进模板。

        当前的实例：EMBEDDING_API_KEY 回退成 LLM_API_KEY（DeepSeek 的 key），
        而 EMBEDDING_API_BASE 写死硅基流动 —— 两家 key 不通用，必然 401。
        只要这个回退还在，模板就必须提醒使用者单独填。
        """
        unsafe = _unsafe_key_fallbacks()
        assert unsafe, "没有跨服务回退的键了？如果确实改好了，请同步更新本测试"
        missing = [name for name in unsafe if name not in EXAMPLE]
        assert not missing, (
            f"这些键缺失时会拿另一家的凭据去请求自家端点（静默走错配置、必然 401），"
            f"必须在 .env.example 里显式给出：{missing}"
        )

    def test_embedding_endpoint_is_documented(self):
        """光有 key 不够——base_url 是另一家，必须一并给出才配得起来。"""
        assert "EMBEDDING_API_KEY" in EXAMPLE
        assert "EMBEDDING_API_BASE" in EXAMPLE


class TestNoAssignmentTraps:
    def test_no_accidental_empty_assignment(self):
        """`KEY=` 会把变量设成空串，而 os.getenv 见到"变量存在"就不再走默认值
        —— 写了比不写更坏。只有语义上"空即有效"的键才允许这样写。
        """
        trapped = [k for k, v in EXAMPLE.items() if v == "" and k not in _EMPTY_ALLOWED]
        assert not trapped, (
            f"这些键被赋成了空串（会覆盖掉代码里的默认值）：{trapped}"
        )

    def test_example_parses_and_is_not_empty(self):
        assert len(EXAMPLE) > 10, f"模板解析出的键太少（{len(EXAMPLE)}），文件可能坏了"


class TestExistingDefaultsMatch:
    def test_default_on_switches_are_documented(self):
        """默认开的功能必须能被人发现并关掉。"""
        missing = [k for k in DEFAULT_ON_SWITCHES if k not in EXAMPLE]
        assert not missing, f"这些默认开启的开关没写进 .env.example：{missing}"

    def test_documented_switch_values_are_parseable_bools(self):
        for name in DEFAULT_ON_SWITCHES:
            if name.endswith("_ENABLED"):
                assert EXAMPLE[name].lower() in ("true", "false"), (
                    f"{name} 的值必须是 true/false 字符串"
                )


class TestLangfuseVariableName:
    """langfuse SDK 只读 LANGFUSE_HOST（v2.50 源码写死这个名字）。

    写成 LANGFUSE_BASE_URL 不会报错，但会被**完全忽略**、静默回落到默认值，
    表现为"改了配置却没生效"。本项目本地 .env 就踩过这个（键名写错，
    只因值恰好等于默认值才没出事）。
    """

    def test_uses_sdk_native_variable_name(self):
        assert "LANGFUSE_HOST" in EXAMPLE

    def test_does_not_advertise_ignored_variable_name(self):
        assert "LANGFUSE_BASE_URL" not in EXAMPLE
