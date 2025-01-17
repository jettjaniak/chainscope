# chainscope

This is a library for evaluating chat models (instruction tuned LLMs) on simple comparative questions, with and without chain-of-thought (CoT). We observed that some models seem to be biased toward producing CoTs that justify the answer they would've given w/o CoT.

Questions are based on a small database of properties of different types of objects in `chainscope/data/properties` and generated programatically by `scripts/gen_qs.py`.
Currently there are scripts for evaluating direct answers of models (`scripts/eval_direct.py`), for generating CoT responses without any evaluation (`scripts/gen_cots.py`), and for extracting the answer from CoT repsonses (`scripts/eval_cots.py`).

## setup instructions
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