from scalesim.memory.buckyball_compiler import compile_gemm_bank_plan


def test_compiler_plan_is_legal_and_contains_static_incumbent():
    for shape in ((256, 96, 96), (32, 432, 96), (50, 96, 384)):
        plan = compile_gemm_bank_plan(*shape)
        allocation = plan.allocation
        assert allocation.total <= 30
        assert allocation.accumulator % 4 == 0 or plan.fallback_used
        assert plan.objective.total_cycles <= plan.static_cycles
