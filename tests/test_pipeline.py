"""
test_pipeline.py — Query Execution Plan（docs 第八章）

resolve_plan 是纯函数：开关不传读 config（默认全 false，CI 无 key 环境安全），
显式传参可覆盖。测试用显式传参驱动各分支，不依赖 monkeypatch config。
"""
import pytest

from src.pipeline import PipelinePlan, resolve_plan


class TestResolvePlan:
    def test_chitchat_all_false_minimal(self):
        """用例 1：chitchat → 全 false，reason=chitchat_minimal，0 次 LLM 调用。"""
        plan = resolve_plan("chitchat",
                            expansion_enabled=True, verify_enabled=True,
                            context_expansion_enabled=True, direct_return_enabled=True)
        assert plan == PipelinePlan(
            use_expansion=False, use_parent_child=False,
            use_crag=False, use_direct_return=False,
            reason="chitchat_minimal", est_llm_calls=0,
        )

    def test_simple_expansion_and_crag_mutually_exclusive(self):
        """用例 2：simple 下扩展和 CRAG 互斥——同开时只保留扩展；可只开一个或都不开。"""
        both_on = resolve_plan("simple", expansion_enabled=True, verify_enabled=True,
                               context_expansion_enabled=False, direct_return_enabled=False)
        assert both_on.use_expansion is True
        assert both_on.use_crag is False
        assert both_on.reason == "simple_expanded"

        verify_only = resolve_plan("simple", expansion_enabled=False, verify_enabled=True,
                                   context_expansion_enabled=False, direct_return_enabled=False)
        assert verify_only.use_crag is True
        assert verify_only.use_expansion is False
        assert verify_only.reason == "simple_verified"

        lean = resolve_plan("simple", expansion_enabled=False, verify_enabled=False,
                            context_expansion_enabled=False, direct_return_enabled=False)
        assert lean.use_expansion is False and lean.use_crag is False
        assert lean.reason == "simple_lean"

    def test_simple_expansion_on_disables_crag(self):
        """用例 3：QUERY_EXPANSION_ENABLED=true 时 use_crag 必须为 False（互斥的另一半）。"""
        plan = resolve_plan("simple", expansion_enabled=True, verify_enabled=True,
                            context_expansion_enabled=False, direct_return_enabled=False)
        assert plan.use_crag is False

    def test_multi_hop_full_pipeline(self):
        """用例 4：multi-hop → 扩展恒关，CRAG 跟随 ANSWER_VERIFY_ENABLED，其余按开关。"""
        verify_on = resolve_plan("multi-hop", expansion_enabled=True, verify_enabled=True,
                                 context_expansion_enabled=True, direct_return_enabled=True)
        assert verify_on.use_expansion is False
        assert verify_on.use_crag is True
        assert verify_on.use_parent_child is True
        assert verify_on.use_direct_return is True
        assert verify_on.reason == "multi_hop_full"
        assert verify_on.est_llm_calls == 3  # 分解 + 生成 + 验证

        verify_off = resolve_plan("multi-hop", expansion_enabled=True, verify_enabled=False,
                                  context_expansion_enabled=False, direct_return_enabled=False)
        assert verify_off.use_crag is False
        assert verify_off.est_llm_calls == 2  # 分解 + 生成

    @pytest.mark.parametrize("query_type", ["chitchat", "simple", "multi-hop"])
    @pytest.mark.parametrize("exp,verify", [(True, True), (True, False), (False, True), (False, False)])
    def test_reason_never_empty(self, query_type, exp, verify):
        """用例 5：所有路径的 reason 非空——日志归因靠它，空串等于归因失效。"""
        plan = resolve_plan(query_type, expansion_enabled=exp, verify_enabled=verify,
                            context_expansion_enabled=False, direct_return_enabled=False)
        assert isinstance(plan.reason, str) and plan.reason
