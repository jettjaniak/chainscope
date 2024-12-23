import pytest

from chainscope.questions import gen_qs


@pytest.mark.parametrize("comparison", ["gt", "lt"])
@pytest.mark.parametrize("answer", ["YES", "NO"])
@pytest.mark.parametrize("prop_id", ["aircraft-speeds", "element-densities"])
def test_gen_qs(comparison, answer, prop_id):
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

    for question in dataset.question_by_qid.values():
        if (answer == "YES" and comparison == "gt") or (
            answer == "NO" and comparison == "lt"
        ):
            assert question.x_value > question.y_value
        else:
            assert question.x_value < question.y_value
