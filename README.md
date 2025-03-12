# ChainScope

This is a library for evaluating chat models (instruction tuned LLMs) on simple comparative questions, with and without chain-of-thought (CoT). We observed that some models seem to be biased toward producing CoTs that justify the answer they would've given w/o CoT.

The repo also includes experiments for evaluating Restoration Errors and Unfaithful Shortcuts (see [our arXiv paper](https://arxiv.org/abs/2503.08679) for definitions, etc) -- exact scripts and data from our paper are **currently being ported** -- sorry for the wait!

Comparative questions are based on a small database of properties of different types of objects in `chainscope/data/properties` and generated programatically by `scripts/gen_qs.py`.
Currently there are scripts for evaluating direct answers of models (`scripts/eval_direct.py`), for generating CoT responses without any evaluation (`scripts/gen_cots.py`), and for extracting the answer from CoT repsonses (`scripts/eval_cots.py`).

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

# Restoration errors and unfaithful shortcuts

Data and scripts for reproducing restoration errors and unfaithful shortcuts will be added to the repo in the coming week (as of March 11 2025).

## `putnamlike` scripts

To evaluate Restoration Errors and Unfaithful Shortcuts, we use the `putnamlike` pipeline of scripts.

1. `putnamlike0_save_rollouts.py`
2. `putnamlike1_are_rollouts_correct.py`
3. `putnamlike2_split_cots.py`
   * WARNING! This is somewhat unreliable, particularly for really long rollouts, as it does only very basic checks of the correct format by checking that the length of the steps added together is within 10% of the original response length. Empirically, using Claude 3.7 Sonnet is much better than other LLMs (far more max output tokens, and not lazy)
4. `putnamlike2p5_critical_steps_eval.py`
   * Optional, reduces number of steps to evaluate
   * If used, then also pass the flag `--critical_steps_yaml=...` to `putnamlike3_main_faithfulness_eval.py`
5. `putnamlike3_main_faithfulness_eval.py`
   * Pass the file output from `putnamlike2_split_cots.py` to this (and possibly `--critical_steps_yaml=...` too, see `putnamlike2p5_critical_steps_eval.py`)

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
