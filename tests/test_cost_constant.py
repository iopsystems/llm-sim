from vllm_sim.cost.constant import ConstantCostModel


def test_returns_fixed_latency_regardless_of_input():
    model = ConstantCostModel(latency_s=0.01)
    assert model.step_latency(scheduler_output=None, state=None) == 0.01
    assert model.step_latency(scheduler_output="anything", state={"x": 1}) == 0.01


def test_is_a_cost_model():
    from vllm_sim.cost.base import CostModel
    model = ConstantCostModel(latency_s=0.02)
    assert isinstance(model, CostModel)
