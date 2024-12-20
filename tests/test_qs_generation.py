import pytest

from chainscope.qs_generation import gen_qs


@pytest.mark.parametrize("comparison", ["gt", "lt"])
@pytest.mark.parametrize("answer", ["YES", "NO"])
@pytest.mark.parametrize("prop_id", ["aircraft-speeds", "element-densities"])
def test_gen_qs_parametrized(comparison, answer, prop_id):
    dataset = gen_qs(
        answer=answer,
        comparison=comparison,
        max_comparisons=3,
        prop_id=prop_id,
    )

    assert len(dataset.question_by_qid) > 0
    assert dataset.prop_id == prop_id
    assert dataset.comparison == comparison
    assert dataset.answer == answer
