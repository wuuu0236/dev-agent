"""
引用来源整理——把检索结果和模型引用，整理成前端能展示、用户能核实、可独立测试的结构

这一层解决两件事：

① **引用不靠模型自觉（防编造）**
   模型只准输出序号 [n]，文件名 / 页码一律由代码映射回真实检索结果。
   模型写 [9] 而上下文只有 3 条 → 直接丢弃，不给它编造文件名的机会。

② **能被核实，才叫引用**
   只显示「文件名 + 页码」其实无法核实——用户看不到原文，也就无从判断模型是在
   照实回答、还是在拿别的内容硬编。所以每条来源都带上命中的原文片段（snippet），
   前端展开即可逐句对照。**「验证」是产品闭环里最容易断掉的一环。**

为什么单独成一个模块：
   这段逻辑原先长在 `src/rag_qa.py` 里，而 rag_qa 依赖 langfuse，导致引用相关的
   单元测试在缺 langfuse 的环境里连收集都做不到——只测纯字符串映射，却要拖上整个
   主链路的重依赖。拎出来之后，这一层可以独立测试。
"""
import re

from src.config import SOURCE_SNIPPET_CHARS


def snippet(content: str, limit: int | None = None) -> str:
    """截取用于展示的原文片段，超长补省略号。

    只做截断、不做清洗：这一层的职责是"让用户看到检索到的东西"，
    任何改写都会让它不再忠实于原文，也就失去了核实的意义。
    """
    text = (content or "").strip()
    n = SOURCE_SNIPPET_CHARS if limit is None else limit
    if n <= 0 or len(text) <= n:
        return text
    return text[:n] + "…"


def _entry(c: dict) -> dict:
    """构造一条展示用来源记录（片段后续追加）。"""
    return {
        "source": c.get("source", ""),
        "page": c.get("page", 0),
        "type": c.get("type", "text"),
        "snippets": [],
    }


def _append_snippet(entry: dict, content: str) -> None:
    snip = snippet(content)
    if snip and snip not in entry["snippets"]:
        entry["snippets"].append(snip)


def unique_sources(contexts: list[dict]) -> list[dict]:
    """按来源去重整理检索结果；同一文档命中多段时，片段合并在该来源下。

    这是「回答没标引用」时的回退展示：模型没给序号，就把检索到的全部来源按文档
    列出来，让用户自己看依据。同一文档的多段内容不丢弃——那正是最该被看到的部分。
    """
    order: list[str] = []
    by_source: dict[str, dict] = {}
    for c in contexts or []:
        if not isinstance(c, dict):
            continue
        src = c.get("source")
        if not src:
            continue
        if src not in by_source:
            by_source[src] = _entry(c)
            order.append(src)
        _append_snippet(by_source[src], c.get("content", ""))
    return [by_source[s] for s in order]


def extract_cited_sources(answer: str, contexts: list[dict]) -> list[dict]:
    """解析回答中的 [n] 引用序号，映射到真实检索来源（按出现顺序、按来源去重）。

    防 LLM 编造：文件名 / 页码不由模型生成，只让它给序号，由本函数映射回检索到的
    真实来源。越界序号、非数字、没引用 —— 一律不产出条目。
    每条来源附带被引用片段的原文，用户展开即可核实。
    """
    if not contexts:
        return []
    order: list[str] = []
    by_source: dict[str, dict] = {}
    for n in re.findall(r"\[(\d+)\]", answer or ""):
        i = int(n) - 1
        if not (0 <= i < len(contexts)):
            continue
        c = contexts[i]
        if not isinstance(c, dict):
            continue
        src = c.get("source")
        if not src:
            continue
        if src not in by_source:
            by_source[src] = _entry(c)
            order.append(src)
        _append_snippet(by_source[src], c.get("content", ""))
    return [by_source[s] for s in order]
