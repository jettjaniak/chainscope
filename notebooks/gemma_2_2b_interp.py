# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float

from chainscope.cot_generation import build_fsp_prompt
from chainscope.typing import *
from chainscope.utils import (get_model_device, load_model_and_tokenizer,
                              make_chat_prompt)

# %%
# Load model and tokenizer
model_id = "google/gemma-2-2b"
fsp_model_id = "google/gemma-2b-it"
# Load model if needed

# %%

# Load the data
df = pd.read_pickle(DATA_DIR / "df-wm-non-ambiguous-hard-2.pkl")
# Columns: q_str, qid, prop_id, comparison, answer, dataset_id, dataset_suffix, model_id, p_yes, p_no, p_correct, mode, instr_id, x_name, y_name, x_value, y_value, temperature, top_p, max_new_tokens, unknown_rate

df = df[df["mode"] == "cot"]
df = df[df["model_id"] == model_id]

assert len(df) > 0, f"No rows found for model {model_id}"

# %%

qid = "03f30cb0ccd65e986ac82df662bae59c095bf89b4658eda25debfeaefeeff34a"

row = df[df["qid"] == qid].iloc[0]
prop_id = row["prop_id"]
dataset_suffix = row["dataset_suffix"]
comparison = row["comparison"]
answer = row["answer"]
dataset_id = row["dataset_id"]

# %% Load faithfulness data and unfaithfulness pattern evaluation data
faithfulness_dataset = UnfaithfulnessPairsDataset.load(model_id, prop_id, dataset_suffix)
unfaithfulness_pattern_eval = UnfaithfulnessPatternEval.load(model_id, prop_id, dataset_suffix)

# %%
prompt = faithfulness_dataset.questions_by_qid[qid].prompt
print(prompt)

# %%
q1_analysis = unfaithfulness_pattern_eval.pattern_analysis_by_qid[qid].q1_analysis
if q1_analysis is None:
    raise ValueError(f"No Q1 analysis found for {qid}")

response_ids_showing_answer_flipping = []
all_response_ids = list(q1_analysis.responses.keys())
for response_id, response_analysis in q1_analysis.responses.items():
    if response_analysis.answer_flipping_classification == "YES":
        response_ids_showing_answer_flipping.append(response_id)

print(f"There are {len(response_ids_showing_answer_flipping)} responses showing answer flipping out of {len(q1_analysis.responses)} total responses")

# %%

# build original FSP
fsp = build_fsp_prompt(
    model_id_for_fsp=fsp_model_id,
    fsp_size=5,
    instr_id="instr-wm",
    ds_params=DatasetParams.from_id(dataset_id),
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=2000,
    ),
    fsp_seed=42,
    instruction_cache={},
    cot_responses_cache={},
    qs_dataset_cache={},
)
print(fsp)
# %%

input_str = f"{fsp}\n\n{prompt}"
