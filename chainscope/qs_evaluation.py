import math
from dataclasses import dataclass
from typing import Literal

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chainscope.qs_generation import QsDataset
from chainscope.utils import make_chat_prompt


@dataclass
class NoCotProbs:
    yes_prob: float
    no_prob: float
    p_correct: float


@dataclass
class NoCotEval:
    probs_by_qid: dict[str, NoCotProbs]
    dataset_id: str
    model_id: str
    seed: int


def logits_to_probs(
    yes_logit: float, no_logit: float, expected_answer: Literal["yes", "no"]
) -> NoCotProbs:
    exp_yes = math.exp(yes_logit)
    exp_no = math.exp(no_logit)
    denom = exp_yes + exp_no
    p_correct = exp_yes / denom if expected_answer == "yes" else exp_no / denom
    return NoCotProbs(
        yes_prob=exp_yes / denom, no_prob=exp_no / denom, p_correct=p_correct
    )


def get_no_cot_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_str: str,
    expected_answer: Literal["yes", "no"],
    prompt: str,
):
    assert not q_str.endswith("Answer:")
    assert "Let's think step by step:" not in q_str
    assert not q_str.startswith("Question: ")

    yes_tok_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_tok_id = tokenizer.encode("No", add_special_tokens=False)[0]

    chat_input_str = make_chat_prompt(
        instruction=prompt.format(q_str=q_str),
        tokenizer=tokenizer,
    )
    input_ids = tokenizer.encode(chat_input_str, add_special_tokens=False)
    logits = model(torch.tensor([input_ids]).cuda()).logits[0, -1]
    yes_logit = logits[yes_tok_id].item()
    no_logit = logits[no_tok_id].item()

    return logits_to_probs(yes_logit, no_logit, expected_answer)


def evaluate_no_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_dataset: QsDataset,
    prompt: str,
    model_id: str,
    dataset_id: str,
    seed: int,
) -> NoCotEval:
    results = {}
    for q in tqdm(question_dataset.questions, desc="Evaluating no-CoT"):
        question = q.q_str
        expected_answer = q.expected_answer
        assert question.endswith("?")
        assert "Question: " not in question
        assert "Let's think step by step:" not in question

        results[q.q_id] = get_no_cot_probs(
            model=model,
            tokenizer=tokenizer,
            question=question,
            expected_answer=expected_answer,
            prompt=prompt,
        )

    return NoCotEval(
        probs_by_qid=results,
        model_id=model_id,
        dataset_id=dataset_id,
        seed=seed,
    )
