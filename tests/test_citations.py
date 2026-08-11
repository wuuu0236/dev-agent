"""确定性测试：RAG 回答的 [n] 引用 → 真实来源映射（不调 LLM，无需任何密钥）。

extract_cited_sources 是「引用不靠模型自觉」的执行层：
模型只给序号 [n]，由代码把序号映射回真实检索来源，越界/编造直接丢弃。
"""
from src.rag_agent import extract_cited_sources


def _ctx(source, page=0, ctype="text"):
    return {"source": source, "page": page, "type": ctype, "content": "..."}


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
