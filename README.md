# ChainScope

This repository contains the datasets and evaluation scripts for the [Chain-of-Thought Reasoning In The Wild Is Not Always Faithful](https://arxiv.org/abs/2503.08679) paper.

## Setup Instructions

1. install python3.12
1. setup your private SSH key
   1. put it under in `.ssh/id_[protocol]`
   1. `chmod 600 [key]`
   1. you can debug with `ssh -T -v git@github.com`
1. clone the repo via ssh `git@github.com:jettjaniak/chainscope.git`
1. make virtual env `python3.12 -m venv .venv`
1. activate virtual env `source .venv/bin/activate`
1. upgrade pip `pip install --upgrade pip`
1. install project in editable state `pip install -e .`
1. install pre-commit hooks `pre-commit install && pre-commit run`
1. run `pytest`

# Implicit Post-Hoc Rationalization

To perform the external consistency analysis in our paper, we generate a dataset of pairs of comparative questions using a subset of the _World Model_ dataset ([Gurnee & Tegmark, 2024](https://arxiv.org/abs/2310.02207)).

1. The subset of properties from this dataset that we use can be found in `chainscope/data/properties`. 
2. We use these properties to generate the questions programatically using `scripts/datasets/gen_qs.py`. The final datasets are in `chainscope/data/questions` (yamls starting with "wm"). A more detailed description of the datasets can be found [here](chainscope/data/questions/datasets.md).
3. We generate CoT responses for the models using `scripts/iphr/gen_cots.py`. The raw responses are in `chainscope/data/cot_responses`.
4.  We evaluate these responses using `scripts/iphr/eval_cots.py`, and the eval results are in `chainscope/data/cot_eval`.

# Restoration Errors

The datasets for Restoration Erros can be found in `chainscope/data/problems`. 
1. We generate responses using `scripts/restoration_errors/gen_cot_paths.py`. The raw responses are in `chainscope/data/cot_paths`. 
2. We evaluate the responses with `scripts/restoration_errors/eval_cot_paths.py`, and the eval results are in `chainscope/data/cot_path_eval`.

# Unfaithful Shortcuts

To evaluate Unfaithful Shortcuts, we use the `putnamlike` pipeline of scripts.

1. `scripts/putnam/putnamlike0_save_rollouts.py`
2. `scripts/putnam/putnamlike1_are_rollouts_correct.py`
3. `scripts/putnam/putnamlike2_split_cots.py`
   * WARNING! This is somewhat unreliable, particularly for really long rollouts, as it does only very basic checks of the correct format by checking that the length of the steps added together is within 10% of the original response length. Empirically, using Claude 3.7 Sonnet is much better than other LLMs (far more max output tokens, and not lazy)
4. `scripts/putnam/putnamlike2p5_critical_steps_eval.py`
   * Optional, reduces number of steps to evaluate
   * If used, then also pass the flag `--critical_steps_yaml=...` to `scripts/putnam/putnamlike3_main_faithfulness_eval.py`
5. `scripts/putnam/putnamlike3_main_faithfulness_eval.py`
   * Pass the file output from `scripts/putnam/putnamlike2_split_cots.py` to this (and possibly `--critical_steps_yaml=...` too, see `scripts/putnam/putnamlike2p5_critical_steps_eval.py`)

# Citation

To cite this work, you can use [our arXiv paper](https://arxiv.org/abs/2503.08679) citation:

```
@misc{arcuschin2025chainofthoughtreasoningwildfaithful,
      title={Chain-of-Thought Reasoning In The Wild Is Not Always Faithful}, 
      author={Iván Arcuschin and Jett Janiak and Robert Krzyzanowski and Senthooran Rajamanoharan and Neel Nanda and Arthur Conmy},
      year={2025},
      eprint={2503.08679},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.08679}, 
}
```
