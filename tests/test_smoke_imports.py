"""冒烟测试：核心模块能正常 import。

抓「导入即崩」类问题——比如 langfuse 不可用时是否正确降级、
依赖缺了会不会在 import 阶段就报错。这是改动后最该先验证的一层。

⚠️ 这里**故意**用 `importlib.import_module("x.y")` 而不是 `import x.y` 语句：
本文件要的正是「导入这个动作本身」当作断言，没有任何名字会被用到。写成 import
语句时，静态检查（pyflakes）只认第一次绑定的 `src`，后面几个会被判成「重复且未
使用」——它读不出「导入即断言」的意图。写成字符串导入，意图就直说了。
（`# noqa` 挡不住：那是 flake8 的功能，pyflakes 不认。）
"""
import importlib

import pytest

MODULES = [
    "src.agent.dev_agent_langgraph",  # LangGraph Agent（含 langfuse 可选降级）
    "src.hybrid_retriever",           # 混合检索
    "src.rag_qa",                     # RAG 问答（@observe 装饰器）
]


@pytest.mark.parametrize("module", MODULES)
def test_core_modules_import(module):
    """逐个导入：一条挂了不影响看到其余两条的结果。"""
    importlib.import_module(module)
