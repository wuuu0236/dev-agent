"""冒烟测试：核心模块能正常 import。

抓「导入即崩」类问题——比如 langfuse 不可用时是否正确降级、
依赖缺了会不会在 import 阶段就报错。这是改动后最该先验证的一层。
"""
def test_core_modules_import():
    import src.agent.dev_agent_langgraph  # LangGraph Agent（含 langfuse 可选降级）
    import src.hybrid_retriever          # 混合检索
    import src.rag_agent                 # RAG 问答（@observe 装饰器）
