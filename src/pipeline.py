"""
pipeline.py — Query Execution Plan（执行计划管线，docs/technical-optimization-plan.md 第八章）

为什么需要：每个模块独立读全局开关，彼此不知道对方跑没跑、花了多少调用。全部
开启时单次提问可能触发 7 次 LLM 调用，且非流式和流式路径的 Parent-Child 扩展
时机曾不一致（非流式在 verify 之后才扩展，CRAG 验证的是小块）。

企业做法：路由决策解析出一条执行计划（PipelinePlan），后续所有步骤只看计划，
不看独立开关。互斥规则收口在一个函数里——simple 类型下扩展和 CRAG 二选一，
互斥逻辑只写这一处，不再散落在各模块。

约定（全局约定第 9 条）：后续所有模块的"该不该跑"由 resolve_plan() 统一决策，
模块内部不再各自读全局开关。

现有 .env 开关不删：它们作为输入传进 resolve_plan()，用户仍然通过环境变量
控制每个模块的全局可用性；同一请求内的实际执行由 plan 决定。

测试约定：纯函数、无 I/O、顶层只 import config 常量——无 API key 环境可 import。
调用方（rag_qa）把自己命名空间的开关值传进来，这样测试 patch rag_qa.XXX 时
计划随之变化，与既有测试的 patch 方式完全兼容（见 tests/test_direct_return.py）。
"""
from dataclasses import dataclass

from src.config import (
    QUERY_EXPANSION_ENABLED,
    CONTEXT_EXPANSION_ENABLED,
    ANSWER_VERIFY_ENABLED,
    DIRECT_RETURN_ENABLED,
)


@dataclass(frozen=True)
class PipelinePlan:
    """一次问答的执行计划（不可变：解析之后不允许执行途中再改主意）。"""

    use_expansion: bool        # Multi-Query 查询扩展（第四章）
    use_parent_child: bool     # Parent-Child 邻域扩展（第三章）
    use_crag: bool             # CRAG 生成后自检（第五章）
    use_direct_return: bool    # 高相似直接返回（第六章）
    reason: str                # 日志归因用 → answer_log.pipeline_reason
    est_llm_calls: int         # 预估 LLM 调用次数（观测用，非精确账单）


def resolve_plan(query_type: str, *,
                 expansion_enabled: bool | None = None,
                 verify_enabled: bool | None = None,
                 context_expansion_enabled: bool | None = None,
                 direct_return_enabled: bool | None = None) -> PipelinePlan:
    """路由结果 → 执行计划。纯函数，可单测。

    四个开关参数不传时读 config（.env）；调用方显式传值可覆盖——rag_qa 传
    自己命名空间的常量，已有测试 patch rag_qa.XXX 的方式照常生效。
    """
    use_exp = QUERY_EXPANSION_ENABLED if expansion_enabled is None else expansion_enabled
    use_verify = ANSWER_VERIFY_ENABLED if verify_enabled is None else verify_enabled
    use_pc = (CONTEXT_EXPANSION_ENABLED if context_expansion_enabled is None
              else context_expansion_enabled)
    use_dr = DIRECT_RETURN_ENABLED if direct_return_enabled is None else direct_return_enabled

    if query_type == "chitchat":
        # 闲聊：不检索不生成大答案，计划全部关——省下扩展、验证、直接返回整条链路。
        return PipelinePlan(
            use_expansion=False, use_parent_child=False,
            use_crag=False, use_direct_return=False,
            reason="chitchat_minimal", est_llm_calls=0,
        )

    if query_type == "multi-hop":
        # 多跳：扩展关（decompose_query 自己拆子问题，再扩展只会重复拆），
        # CRAG / Parent-Child / 直接返回按各自全局开关全开。
        return PipelinePlan(
            use_expansion=False,
            use_parent_child=use_pc,
            use_crag=use_verify,
            use_direct_return=use_dr,
            reason="multi_hop_full",
            est_llm_calls=2 + int(use_verify),  # 分解 + 生成（+ 验证）
        )

    # simple：扩展和 CRAG 二选一（互斥，二者都开会让验证对象变得含混且账单翻倍），
    # 都不开也行。互斥规则只在这一处写。
    use_crag = use_verify and not use_exp
    return PipelinePlan(
        use_expansion=use_exp,
        use_parent_child=use_pc,
        use_crag=use_crag,
        use_direct_return=use_dr,
        reason="simple_expanded" if use_exp else ("simple_verified" if use_crag else "simple_lean"),
        est_llm_calls=1 + int(use_exp) + int(use_crag),  # 生成（+ 扩展）（+ 验证）
    )
