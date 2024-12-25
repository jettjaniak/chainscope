import math

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from chainscope.questions import QsDataset
from chainscope.typing import *
from chainscope.utils import get_model_device, make_chat_prompt


def logits_to_probs(yes_logit: float, no_logit: float) -> DirectEvalProbs:
    exp_yes = math.exp(yes_logit)
    exp_no = math.exp(no_logit)
    denom = exp_yes + exp_no
    return DirectEvalProbs(
        p_yes=exp_yes / denom,
        p_no=exp_no / denom,
    )


def get_direct_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_str: str,
    direct_instruction: str,
):
    device = get_model_device(model)

    yes_tok_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_tok_id = tokenizer.encode("NO", add_special_tokens=False)[0]

    instruction_template = direct_instruction
    # TODO: check if this contains BOS for models that need it when needed
    chat_input_str = make_chat_prompt(
        instruction=instruction_template.format(question=question_str),
        tokenizer=tokenizer,
    )
    input_ids = tokenizer.encode(chat_input_str, add_special_tokens=False)
    logits = model(torch.tensor([input_ids]).to(device)).logits[0, -1]
    yes_logit = logits[yes_tok_id].item()
    no_logit = logits[no_tok_id].item()

    return logits_to_probs(yes_logit, no_logit)


def evaluate_direct(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_dataset: QsDataset,
    instr_id: str,
) -> DirectEval:
    instructions = Instructions.load(instr_id)
    results = {}
    for qid, q in tqdm(
        question_dataset.question_by_qid.items(), desc="Evaluating direct responses"
    ):
        results[qid] = get_direct_probs(
            model=model,
            tokenizer=tokenizer,
            question_str=q.q_str,
            direct_instruction=instructions.direct,
        )

    return DirectEval(
        probs_by_qid=results,
        model_id=model.name_or_path,
        instr_id=instr_id,
        ds_params=question_dataset.params,
    )
