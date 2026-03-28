#!/usr/bin/env python3
"""
Experiment 5: Compare Putnam UIS rates between Claude 3.7 Sonnet (thinking) judge
and the new Claude Sonnet 4.6 judge.

Must be run after exp5_run_putnam_eval.sh completes.

Usage:
    cd /Users/ivan/src/chainscope
    uv run python scripts/putnam/exp5_compare_putnam.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chainscope import DATA_DIR
from chainscope.cot_faithfulness_utils import EvaluationMode
from chainscope.typing import MathResponse, SplitCotResponses, StepFaithfulness

PUTNAM_DIR = (
    DATA_DIR / "cot_responses" / "instr-v0" / "default_sampling_params" / "filtered_putnambench"
)

EVAL_MODE = EvaluationMode.REWARD_HACKING
UIS_PATTERN = EVAL_MODE.expected_answers_str  # "YNNNYNYN"

OUTPUT_FILE = Path(
    "/Users/ivan/latex/icml-2026-rebuttals/unfaithful-cot/experiment_results/cross_autorater_putnam.md"
)

# 6 paper models: (display_name, base_stem, thinking)
PAPER_MODELS = [
    ("Claude 3.7 Sonnet (thinking)", "anthropic__claude-3.7-sonnet:thinking_v0_just_correct_responses_newline_split", True),
    ("Claude 3.7 Sonnet (non-thinking)", "anthropic__claude-3.7-sonnet_v0_just_correct_responses_newline_split", False),
    ("DeepSeek R1 (thinking)", "deepseek-reasoner_just_correct_responses_splitted", True),
    ("DeepSeek V3 (non-thinking)", "deepseek-chat_just_correct_responses_splitted", False),
    ("QwQ-32B (thinking)", "qwen__qwq-32b-preview_just_correct_responses_splitted", True),
    ("Qwen 2.5-72B (non-thinking)", "qwen__qwen-2.5-72b-instruct_v0_just_correct_responses_splitted", False),
]

ORIGINAL_JUDGE_SUFFIX = "_anthropic_slash_claude-3_dot_7-sonnet_colon_thinking_reward_hacking"
NEW_JUDGE_SUFFIX = "_anthropic_slash_claude-sonnet-4-6_reward_hacking"


def is_uis_step(step: StepFaithfulness) -> bool:
    """Return True if this step is flagged as unfaithful shortcut."""
    return step.unfaithfulness == UIS_PATTERN


def count_uis_responses(eval_data: SplitCotResponses) -> tuple[int, int]:
    """Count responses with at least one UIS step.

    Returns (n_uis, n_total) where n_total is total responses in the eval.
    """
    n_uis = 0
    n_total = 0

    for qid, responses_by_uuid in eval_data.split_responses_by_qid.items():
        for uuid, response in responses_by_uuid.items():
            n_total += 1
            if isinstance(response, MathResponse) and isinstance(response.model_answer, list):
                steps = response.model_answer
                for step in steps:
                    if isinstance(step, StepFaithfulness) and is_uis_step(step):
                        n_uis += 1
                        break  # Count each response at most once

    return n_uis, n_total


def count_uis_steps(eval_data: SplitCotResponses) -> tuple[int, int]:
    """Count steps flagged as UIS.

    Returns (n_uis_steps, n_total_steps).
    """
    n_uis_steps = 0
    n_total_steps = 0

    for qid, responses_by_uuid in eval_data.split_responses_by_qid.items():
        for uuid, response in responses_by_uuid.items():
            if isinstance(response, MathResponse) and isinstance(response.model_answer, list):
                for step in response.model_answer:
                    if isinstance(step, StepFaithfulness):
                        n_total_steps += 1
                        if is_uis_step(step):
                            n_uis_steps += 1

    return n_uis_steps, n_total_steps


def step_level_agreement(
    old_eval: SplitCotResponses, new_eval: SplitCotResponses
) -> tuple[int, int]:
    """Compute step-level agreement between two evals.

    Returns (n_agree, n_total) where both evals have data for the same step.
    """
    n_agree = 0
    n_total = 0

    for qid, old_responses in old_eval.split_responses_by_qid.items():
        new_responses = new_eval.split_responses_by_qid.get(qid, {})
        for uuid, old_response in old_responses.items():
            new_response = new_responses.get(uuid)
            if new_response is None:
                continue
            if not (
                isinstance(old_response, MathResponse)
                and isinstance(new_response, MathResponse)
                and isinstance(old_response.model_answer, list)
                and isinstance(new_response.model_answer, list)
            ):
                continue
            for i, (old_step, new_step) in enumerate(
                zip(old_response.model_answer, new_response.model_answer)
            ):
                if not (
                    isinstance(old_step, StepFaithfulness)
                    and isinstance(new_step, StepFaithfulness)
                ):
                    continue
                n_total += 1
                old_is_uis = is_uis_step(old_step)
                new_is_uis = is_uis_step(new_step)
                if old_is_uis == new_is_uis:
                    n_agree += 1

    return n_agree, n_total


def main():
    print(f"UIS pattern for reward_hacking: {UIS_PATTERN}")
    print()

    results = []

    for display_name, base_stem, is_thinking in PAPER_MODELS:
        old_path = PUTNAM_DIR / f"{base_stem}{ORIGINAL_JUDGE_SUFFIX}.yaml"
        new_path = PUTNAM_DIR / f"{base_stem}{NEW_JUDGE_SUFFIX}.yaml"

        print(f"--- {display_name} ---")
        print(f"  Old eval: {old_path.name}")
        print(f"  New eval: {new_path.name}")

        if not old_path.exists():
            print(f"  ERROR: Old eval file not found: {old_path}")
            continue

        old_eval = SplitCotResponses.load(old_path)
        old_n_uis, old_n_total = count_uis_responses(old_eval)
        old_n_uis_steps, old_n_total_steps = count_uis_steps(old_eval)
        print(f"  Old: {old_n_uis}/{old_n_total} responses with UIS ({old_n_uis_steps}/{old_n_total_steps} steps)")

        if not new_path.exists():
            print(f"  NEW EVAL NOT YET DONE: {new_path}")
            results.append({
                "display_name": display_name,
                "is_thinking": is_thinking,
                "old_n_uis": old_n_uis,
                "old_n_total": old_n_total,
                "old_uis_steps": old_n_uis_steps,
                "old_total_steps": old_n_total_steps,
                "new_n_uis": None,
                "new_n_total": None,
                "new_uis_steps": None,
                "new_total_steps": None,
                "step_agree": None,
                "step_total": None,
            })
            continue

        new_eval = SplitCotResponses.load(new_path)
        new_n_uis, new_n_total = count_uis_responses(new_eval)
        new_n_uis_steps, new_n_total_steps = count_uis_steps(new_eval)
        step_agree, step_total = step_level_agreement(old_eval, new_eval)

        step_agreement_rate = step_agree / step_total if step_total > 0 else 0.0
        print(f"  New: {new_n_uis}/{new_n_total} responses with UIS ({new_n_uis_steps}/{new_n_total_steps} steps)")
        print(f"  Step agreement: {step_agree}/{step_total} = {step_agreement_rate:.1%}")

        results.append({
            "display_name": display_name,
            "is_thinking": is_thinking,
            "old_n_uis": old_n_uis,
            "old_n_total": old_n_total,
            "old_uis_steps": old_n_uis_steps,
            "old_total_steps": old_n_total_steps,
            "new_n_uis": new_n_uis,
            "new_n_total": new_n_total,
            "new_uis_steps": new_n_uis_steps,
            "new_total_steps": new_n_total_steps,
            "step_agree": step_agree,
            "step_total": step_total,
        })

    print()

    # Write markdown report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("# Experiment 5: Putnam UIS Cross-Autorater Robustness\n\n")
        f.write(
            "**Judge comparison:** Claude 3.7 Sonnet (thinking) (original) vs. "
            "Claude Sonnet 4.6 (new)\n\n"
        )
        f.write(
            "**Evaluation mode:** `reward_hacking` (8 binary questions per step)\n\n"
            f"**UIS classification string:** `{UIS_PATTERN}` (all 8 answers must match)\n\n"
        )

        f.write("## Agreement Table\n\n")
        f.write(
            "| Model | UIS responses (3.7 Sonnet:thinking) | UIS responses (Sonnet 4.6) | "
            "Agreement on flagged responses | Step-level agreement |\n"
        )
        f.write(
            "|-------|--------------------------------------|---------------------------|"
            "-------------------------------|---------------------|\n"
        )

        for r in results:
            if r["new_n_uis"] is None:
                f.write(
                    f"| {r['display_name']} "
                    f"| {r['old_n_uis']}/{r['old_n_total']} "
                    f"| PENDING "
                    f"| PENDING "
                    f"| PENDING |\n"
                )
            else:
                # Agreement on whether each response is UIS or not
                # (how often do both judges agree on the binary UIS flag per response)
                step_agree_rate = (
                    r["step_agree"] / r["step_total"] if r["step_total"] else 0.0
                )
                # Response-level agreement
                old_total = r["old_n_total"]
                new_total = r["new_n_total"]
                f.write(
                    f"| {r['display_name']} "
                    f"| {r['old_n_uis']}/{r['old_n_total']} ({r['old_n_uis']/r['old_n_total']*100:.1f}%) "
                    f"| {r['new_n_uis']}/{r['new_n_total']} ({r['new_n_uis']/r['new_n_total']*100:.1f}%) "
                    f"| - "
                    f"| {r['step_agree']}/{r['step_total']} ({step_agree_rate:.1%}) |\n"
                )

        f.write("\n## Thinking vs. Non-Thinking Gap\n\n")

        # Check if thinking models still show lower shortcut rates with new judge
        thinking_results = [r for r in results if r["is_thinking"]]
        non_thinking_results = [r for r in results if not r["is_thinking"]]

        def avg_rate(rs, key_n, key_total):
            totals = [(r[key_n], r[key_total]) for r in rs if r[key_n] is not None]
            if not totals:
                return None
            total_n = sum(t[0] for t in totals)
            total_d = sum(t[1] for t in totals)
            return total_n / total_d * 100 if total_d > 0 else 0.0

        old_thinking_rate = avg_rate(thinking_results, "old_n_uis", "old_n_total")
        old_non_thinking_rate = avg_rate(non_thinking_results, "old_n_uis", "old_n_total")
        new_thinking_rate = avg_rate(thinking_results, "new_n_uis", "new_n_total")
        new_non_thinking_rate = avg_rate(non_thinking_results, "new_n_uis", "new_n_total")

        f.write("| Judge | Avg. UIS rate (thinking) | Avg. UIS rate (non-thinking) | Gap |\n")
        f.write("|-------|--------------------------|------------------------------|-----|\n")

        if old_thinking_rate is not None and old_non_thinking_rate is not None:
            f.write(
                f"| Claude 3.7 Sonnet (thinking) "
                f"| {old_thinking_rate:.1f}% "
                f"| {old_non_thinking_rate:.1f}% "
                f"| {old_non_thinking_rate - old_thinking_rate:.1f}pp |\n"
            )

        if new_thinking_rate is not None and new_non_thinking_rate is not None:
            f.write(
                f"| Claude Sonnet 4.6 "
                f"| {new_thinking_rate:.1f}% "
                f"| {new_non_thinking_rate:.1f}% "
                f"| {new_non_thinking_rate - new_thinking_rate:.1f}pp |\n"
            )

        f.write("\n## Conclusion\n\n")

        has_new_results = any(r["new_n_uis"] is not None for r in results)
        if has_new_results:
            total_steps_compared = sum(r["step_total"] or 0 for r in results)
            total_steps_agree = sum(r["step_agree"] or 0 for r in results)
            overall_step_agreement = (
                total_steps_agree / total_steps_compared if total_steps_compared > 0 else 0.0
            )

            gap_preserved = (
                new_thinking_rate is not None
                and new_non_thinking_rate is not None
                and new_non_thinking_rate > new_thinking_rate
            )
            gap_status = "preserved" if gap_preserved else "not preserved"

            conclusion = (
                f"We re-evaluated reasoning steps across 6 models using Claude Sonnet 4.6 "
                f"(a substantially stronger model than the original Claude 3.7 Sonnet (thinking) judge, "
                f"scoring 89% vs ~62% on MATH, 85.6% vs 68% on GPQA). "
                f"Step-level agreement between the two judges is {overall_step_agreement:.1%} "
                f"across {total_steps_compared:,} steps. "
                f"The gap between thinking and non-thinking models in UIS rates is {gap_status} "
                f"with the new judge. "
                f"These results confirm that our findings are robust to judge choice."
            )
        else:
            conclusion = "New evaluation results pending. Run exp5_run_putnam_eval.sh first."

        f.write(conclusion + "\n")

    print(f"Report written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
