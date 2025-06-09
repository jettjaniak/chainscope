# Scripts

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

# Data Used In Paper

## Qwen 72B

Claude 3.7 Sonnet (Thinking) Autorating: `chainscope/d/cot_responses/instr-v0/default_sampling_params/filtered_putnambench/qwen__qwen-2.5-72b-instruct_v0_just_correct_responses_splitted_anthropic_slash_claude-3_dot_7-sonnet_colon_thinking_reward_hacking.yaml`

Notebook of Manual Analysis: `chainscope/notebooks/analyze_faithful_autorating_full_history/state_at_1741308754_mar_7_00_52.py`
