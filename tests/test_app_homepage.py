"""首页文案与代码的一致性守卫。

为什么需要这个文件：首页文案是"说给面试官听的话"，而它已经出过两次
「文案与代码不符」——这类 bug 编译器、常规单测、CI 全都拦不住：

  1. 首页写着「LLM-as-Judge 四维评估」，但评估层早已换成 RAGAS（2026-09-14 修）；
  2. 侧边栏把「LangGraph Agent」「私有化离线（Ollama）」列进**技术栈**，
     但这两条线上 Demo 演示不了、且一个字没标注 → 面试时"以为有、打开没有"。

两次都只能靠人肉发现。本文件把"首页说的话"与"代码实际做的事"绑成断言。

两类检查：
  A. 文本一致性 —— 只依赖 src.config 与纯文本读取，**CI 可跑**（CI 刻意不装
     streamlit / ragas / chromadb，见 requirements-dev.txt）。
     关键设计：不断言"字面量出现过"，而是**从代码里取真实值再对照**——
     例如从 evaluation_ragas.py 读 METRIC_NAMES、从 config 读 BM25_WEIGHT，
     改代码而忘改文案时会立刻失败。
  B. 渲染冒烟 —— 需 streamlit，CI 里 skip、本地跑。
     能拦住"编译通过但 markdown 表格写坏、打开乱版"，这是 A 类做不到的。
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "app.py").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

# ---- 被声明为"仅本机运行"的四项能力（技术栈清单里不得出现）----
LOCAL_ONLY_MARKERS = ["LangGraph", "MCP", "FastAPI", "Ollama"]


# ============================================================
# A 类：文本与代码的一致性（CI 可跑）
# ============================================================

class TestNoStaleWording:
    """已废弃的口径不得回流。"""

    def test_llm_as_judge_wording_absent(self):
        """LLM-as-Judge 是旧口径，主评估已是 RAGAS。"""
        lowered = APP.lower()
        assert "llm-as-judge" not in lowered
        assert "llm 评估面板" not in lowered
        assert "llm 量化数据" not in lowered

    def test_langgraph_not_claimed_as_plain_tech_stack(self):
        """技术栈清单里不得裸列 LangGraph —— Web 四页走的是确定性管道，不经过它。"""
        assert "LangGraph Agent（HTTP API 入口）" not in APP


class TestMetricsMatchCode:
    """首页列的指标名必须与 evaluation_ragas.py 的真实定义一致。

    不写死字面量：从代码里读出 METRIC_NAMES，再换算成展示名去比对。
    这样"改了指标却没改文案"会被直接拦住。
    """

    @staticmethod
    def _real_metric_names() -> list[str]:
        src = (ROOT / "src" / "evaluation_ragas.py").read_text(encoding="utf-8")
        m = re.search(r"METRIC_NAMES\s*=\s*\[(.*?)\]", src, re.S)
        assert m, "evaluation_ragas.py 里找不到 METRIC_NAMES 定义"
        return re.findall(r"\"([a-z_]+)\"", m.group(1))

    def test_homepage_lists_every_real_metric(self):
        names = self._real_metric_names()
        assert len(names) == 4, f"预期四个指标，实际 {names}"
        for name in names:
            display = name.replace("_", " ").title()
            assert display in APP, f"首页缺少指标展示名：{display}"

    def test_context_qualifier_not_dropped(self):
        """Context Recall / Context Precision 的 Context 限定不能丢。

        丢了会与 Answer Relevancy 混淆——前者测"该召回的召回了吗"，
        后者测"答得切不切题"，面试被追问时说不清。
        """
        assert "Recall / Precision" not in APP
        assert "Context Recall" in APP
        assert "Context Precision" in APP


class TestClaimsMatchConfig:
    """首页宣称的每一个能力开关，都必须真的开着。

    这是本文件最值钱的一组——「文案说 A、代码做 B」正是之前踩过的坑：
    首页与 README 都写着"混合检索（BM25 + 向量）"，而 BM25_WEIGHT=0
    让 _get_bm25() 首行短路，实际只有向量单路。
    """

    def test_bm25_claim_is_true(self):
        from src import config

        assert "混合检索" in APP
        assert config.BM25_WEIGHT > 0, "首页说混合检索，但 BM25_WEIGHT=0（实为纯向量单路）"
        assert config.VECTOR_WEIGHT > 0

    def test_hybrid_short_circuit_uses_real_weight(self):
        """_get_bm25 的短路条件必须引用 BM25_WEIGHT，不能写死。

        写死会让"调权重启用混合检索"这个说法失效——那正是它当年
        A/B 完就再也无法复测的原因。
        """
        src = (ROOT / "src" / "hybrid_retriever.py").read_text(encoding="utf-8")
        body = src[src.index("def _get_bm25"):]
        body = body[: body.index("\ndef ") if "\ndef " in body else len(body)]
        assert "BM25_WEIGHT" in body

    def test_gate_claim_is_true(self):
        from src import config

        assert "质量门控" in APP
        assert config.RETRIEVAL_MIN_SCORE > 0, "首页说质量门控，但阈值为 0（门控关闭）"

    def test_rerank_claim_is_true(self):
        from src import config

        assert "精排" in APP
        assert config.RERANK_ENABLED, "首页说交叉编码精排，但 RERANK_ENABLED=False"

    def test_ocr_claim_is_true(self):
        from src import config

        assert "OCR" in APP
        assert hasattr(config, "ENABLE_OCR")


class TestClaimsMatchCode:
    """没有开关的能力，直接查实现是否真的存在。"""

    def test_citation_verifiability_exists(self):
        """首页说"展开可见命中原文"，就必须真有出片段的函数。"""
        assert "引用可核实" in APP
        src = (ROOT / "src" / "citations.py").read_text(encoding="utf-8")
        assert "def extract_cited_sources" in src
        assert "def snippet" in src

    def test_docx_table_parsing_exists(self):
        """首页说"Word（含表格）"，就必须真的读了 doc.tables。"""
        assert "含表格" in APP
        src = (ROOT / "src" / "parser.py").read_text(encoding="utf-8")
        assert "_table_to_text" in src
        assert "doc.element.body" in src, "必须按 body 真实顺序迭代，否则表格会被挪到文末"


class TestEntryBoundaryDeclared:
    """三条入口的线上/本机边界必须写在产品里，而不是只存在于记忆里。

    这是 2026-09-14 永健"面试时搞混了"的直接产物：他记得有某功能，
    打开线上 Demo 却找不到——因为首页把只在本机的能力列进了技术栈。
    """

    def test_tech_stack_lists_only_online_capabilities(self):
        m = re.search(r"### 技术栈(.*?)### ", APP, re.S)
        assert m, "app.py 里找不到「技术栈」分组"
        stack = m.group(1)
        for marker in LOCAL_ONLY_MARKERS:
            assert marker.lower() not in stack.lower(), (
                f"「{marker}」只在本地运行，不得列进技术栈清单——"
                f"会让读者以为线上 Demo 有它"
            )

    def test_local_only_section_exists_in_app(self):
        assert "本机能力" in APP
        for marker in LOCAL_ONLY_MARKERS:
            assert marker in APP, f"首页未说明「{marker}」仅本机"

    def test_every_local_entry_marked_in_app(self):
        # 从 subheader 锚起：这几个字在侧边栏先出现过一次（"原因见首页…"），
        # 直接搜关键词会匹配到侧边栏，取到的是错误区段。
        m = re.search(
            r'st\.subheader\("🧭 运行方式与能力边界"\)(.*?)st\.divider\(\)', APP, re.S
        )
        assert m, "app.py 里找不到「运行方式与能力边界」区"
        section = m.group(1)
        assert section.count("需本机运行") >= 3
        assert "即当前页面" in section

    def test_boundary_is_framed_as_design_not_todo(self):
        """必须点明是"设计决定"——否则读者会当成未完成项来催。"""
        assert "设计决定" in APP

    def test_readme_marks_local_only_entries(self):
        assert "仅在本机运行" in README
        assert "❌ 本机" in README
        assert "MCP 工具服务" in README


# ============================================================
# B 类：渲染冒烟（需 streamlit；CI 里 skip，本地跑）
# ============================================================

def _has_streamlit() -> bool:
    """探测 streamlit 是否可用。

    不用 pytest.importorskip：它在 fixture 里抛的是 error 而非 skip
    （allow_module_level 的 Skipped 在 fixture 上下文中不被当跳过处理）。
    find_spec 只查不导入，快且不触发副作用。
    """
    import importlib.util

    try:
        return importlib.util.find_spec("streamlit") is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="module")
def rendered_app():
    if not _has_streamlit():
        pytest.skip("CI 不装 streamlit（见 requirements-dev.txt），本组仅本地跑")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    return at


class TestHomepageRenders:
    """文案改动写不了常规单测：markdown 表格竖线写坏、缩进写坏，
    编译全过，但一打开就是乱版。只有真渲染一遍才拦得住。
    """

    def test_renders_without_exception(self, rendered_app):
        assert not rendered_app.exception, f"首页渲染抛异常：{rendered_app.exception}"

    def test_tables_have_consistent_columns(self, rendered_app):
        """表格每行竖线数必须一致，否则 markdown 表格会塌成纯文本。"""
        for i, md in enumerate(rendered_app.markdown):
            lines = [ln for ln in md.value.splitlines() if ln.strip().startswith("|")]
            if len(lines) < 2:
                continue
            widths = {ln.count("|") for ln in lines}
            assert len(widths) == 1, f"第 {i} 个 markdown 块的表格列数不一致：{widths}"

    def test_sidebar_separates_local_only(self, rendered_app):
        sidebar = "\n".join(m.value for m in rendered_app.sidebar.markdown)
        assert "本机能力（本 Demo 不含）" in sidebar
        assert "LangGraph Agent（HTTP API 入口）" not in sidebar
