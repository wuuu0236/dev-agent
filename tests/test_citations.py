"""引用来源整理（src/citations.py）单元测试——不调 LLM、无需任何密钥。

这一层是「引用不靠模型自觉」的执行层，做两件事：
  ① 模型只给序号 [n]，文件名/页码由代码映射回真实检索来源，越界/编造直接丢弃；
  ② 每条来源带上命中原文（snippet），让用户能展开核实——引用只给文件名是没法核实的。

注：本测试从 `src.citations` 导入（而非 `src.rag_qa`）——引用逻辑与主链路解耦后，
这一层不再被 langfuse 依赖绑架，可以在任何环境里跑。
"""
from src.citations import extract_cited_sources, unique_sources, snippet
import src.citations as ct


def _ctx(source, page=0, ctype="text", content="..."):
    return {"source": source, "page": page, "type": ctype, "content": content}


# ---------- 序号映射（防编造） ----------

def test_maps_numbers_to_real_sources_dedup():
    """[1]/[3] 映射到真实来源，重复 [1] 只出现一次，按出现顺序。"""
    contexts = [_ctx("a.md", 1), _ctx("b.md", 2), _ctx("c.png", 0, "image")]
    cited = extract_cited_sources("根据 [1] 和 [3] 可知，重复的 [1] 也支持", contexts)
    assert [c["source"] for c in cited] == ["a.md", "c.png"]
    assert cited[0]["page"] == 1
    assert cited[1]["type"] == "image"


def test_ignores_out_of_range_and_fabricated():
    """越界序号 / 非数字 / 完全没引用 → 不映射出任何来源（防编造文件名）。"""
    contexts = [_ctx("a.md")]
    assert extract_cited_sources("提到 [9] 和 [abc]", contexts) == []
    assert extract_cited_sources("没有引用任何文档", contexts) == []


def test_empty_contexts_safe():
    """没检索到上下文时调用也不报错。"""
    assert extract_cited_sources("根据 [1] 引用", []) == []


def test_keeps_legacy_fields():
    """老字段必须保留：缓存里存的旧格式、以及前端旧逻辑还认它们。"""
    cited = extract_cited_sources("[1]", [_ctx("a.md", 3, "image")])
    assert {"source", "page", "type", "snippets"} <= set(cited[0])


def test_cited_groups_multiple_chunks_of_same_source():
    """同一文档被引用到多段 → 合并为一条来源，片段都保留（不能只留第一段）。"""
    contexts = [_ctx("a.md", content="A1"), _ctx("a.md", content="A2")]
    cited = extract_cited_sources("[1] 和 [2]", contexts)
    assert len(cited) == 1
    assert cited[0]["snippets"] == ["A1", "A2"]


def test_cited_dedups_identical_snippets():
    """同一段内容被 [1][2] 重复引用时，片段不重复出现。"""
    contexts = [_ctx("a.md", content="同样的内容"), _ctx("a.md", content="同样的内容")]
    cited = extract_cited_sources("[1][2]", contexts)
    assert cited[0]["snippets"] == ["同样的内容"]


def test_cited_skips_dirty_entries():
    """contexts 里混入脏数据不崩。"""
    contexts = [_ctx("a.md"), "脏", None, {"page": 1}]
    cited = extract_cited_sources("[1][2][3][4]", contexts)
    assert [c["source"] for c in cited] == ["a.md"]


def test_cited_empty_content_yields_source_without_snippet():
    """没有正文的块（理论上不该有）仍给出来源，只是没有片段可展示。"""
    cited = extract_cited_sources("[1]", [_ctx("a.md", content="")])
    assert cited[0]["source"] == "a.md"
    assert cited[0]["snippets"] == []


# ---------- unique_sources（回答没标引用时的回退展示） ----------

def test_unique_sources_groups_snippets_by_source():
    contexts = [
        _ctx("a.md", 1, content="第一段"),
        _ctx("b.md", 0, content="b 的段落"),
        _ctx("a.md", 1, content="第二段"),
    ]
    out = unique_sources(contexts)
    assert [s["source"] for s in out] == ["a.md", "b.md"]      # 按首次出现顺序
    assert out[0]["page"] == 1
    assert out[0]["snippets"] == ["第一段", "第二段"]


def test_unique_sources_skips_dirty_and_missing_source():
    out = unique_sources(["脏", None, {"page": 1}, _ctx("a.md")])
    assert [s["source"] for s in out] == ["a.md"]


def test_unique_sources_empty_input():
    assert unique_sources([]) == []
    assert unique_sources(None) == []


# ---------- snippet 截断 ----------

def test_snippet_truncates_long_text():
    out = snippet("字" * 500)
    assert out.endswith("…")
    assert len(out) == ct.SOURCE_SNIPPET_CHARS + 1


def test_snippet_keeps_short_text_intact():
    assert snippet("短文本") == "短文本"


def test_snippet_limit_override():
    assert snippet("字" * 100, limit=10) == "字" * 10 + "…"


def test_snippet_limit_zero_disables_truncation():
    assert snippet("字" * 100, limit=0) == "字" * 100


def test_snippet_strips_surrounding_whitespace():
    assert snippet("  abc  ") == "abc"


def test_snippet_handles_none():
    assert snippet(None) == ""
