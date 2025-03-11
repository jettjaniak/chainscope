# %%

from chainscope.typing import (
    DATA_DIR,
    AnswerCorrectnessResult,
    CoTPath,
    CoTPathEval,
    FirstPassNodeStatus,
    FirstPassResponseResult,
    SecondPassNodeStatus,
    SecondPassResponseResult,
    ThirdPassNodeStatus,
    ThirdPassResponseResult,
)

# %%

eval_files = list(DATA_DIR.glob("cot_path_eval/**/*.yaml"))
print(f"Processing {len(eval_files)} files")

for eval_file in eval_files:
    print(f"Processing {eval_file}")
    cot_path_eval = CoTPathEval.load_from_path(eval_file)
    cot_path = CoTPath.load(cot_path_eval.model_id, cot_path_eval.problem_dataset_name)

    for qid, response_dict in cot_path.cot_path_by_qid.items():
        if qid not in cot_path_eval.answer_correct_by_qid:
            cot_path_eval.answer_correct_by_qid[qid] = {}
            print(f"Adding empty dict to answer_correct_by_qid for {qid}")
        if qid not in cot_path_eval.first_pass_eval_by_qid:
            cot_path_eval.first_pass_eval_by_qid[qid] = {}
            print(f"Adding empty dict to first_pass_eval_by_qid for {qid}")

        for response_uuid in response_dict.keys():
            all_steps_count = len(cot_path.cot_path_by_qid[qid][response_uuid])

            if response_uuid not in cot_path_eval.answer_correct_by_qid[qid]:
                print(
                    f"Adding UNKNOWN to answer_correct_by_qid for {qid} {response_uuid}"
                )
                cot_path_eval.answer_correct_by_qid[qid][response_uuid] = (
                    AnswerCorrectnessResult(
                        answer_correct="UNKNOWN",
                        explanation="",
                    )
                )
            if response_uuid not in cot_path_eval.first_pass_eval_by_qid[qid]:
                print(
                    f"Adding empty dict to first_pass_eval_by_qid for {qid} {response_uuid}"
                )
                cot_path_eval.first_pass_eval_by_qid[qid][response_uuid] = (
                    FirstPassResponseResult(steps_status={})
                )

            for step_num in cot_path.cot_path_by_qid[qid][response_uuid].keys():
                if (
                    step_num
                    not in cot_path_eval.first_pass_eval_by_qid[qid][
                        response_uuid
                    ].steps_status
                ):
                    print(
                        f"Adding UNKNOWN status to first_pass_eval_by_qid for {qid} {response_uuid} {step_num}"
                    )
                    cot_path_eval.first_pass_eval_by_qid[qid][
                        response_uuid
                    ].steps_status[step_num] = FirstPassNodeStatus(
                        node_status="UNKNOWN", explanation=""
                    )

            has_correct_answer = (
                cot_path_eval.answer_correct_by_qid[qid][response_uuid].answer_correct
                == "CORRECT"
            )

            has_incorrect_steps_in_first_pass = (
                any(
                    step_eval.node_status == "INCORRECT"
                    for step_eval in cot_path_eval.first_pass_eval_by_qid[qid][
                        response_uuid
                    ].steps_status.values()
                )
                and len(
                    cot_path_eval.first_pass_eval_by_qid[qid][
                        response_uuid
                    ].steps_status
                )
                > 0
            )

            if has_correct_answer and has_incorrect_steps_in_first_pass:
                if qid not in cot_path_eval.second_pass_eval_by_qid:
                    cot_path_eval.second_pass_eval_by_qid[qid] = {}
                    print(f"Adding empty dict to second_pass_eval_by_qid for {qid}")

                if response_uuid not in cot_path_eval.second_pass_eval_by_qid[qid]:
                    print(
                        f"Adding empty dict to second_pass_eval_by_qid for {qid} {response_uuid}"
                    )
                    cot_path_eval.second_pass_eval_by_qid[qid][response_uuid] = (
                        SecondPassResponseResult(steps_status={}, reasoning=None)
                    )

                all_nodes_have_status_none_in_second_pass = (
                    all(
                        step_status.node_status == "NONE"
                        for step_status in cot_path_eval.second_pass_eval_by_qid[qid][
                            response_uuid
                        ].steps_status.values()
                    )
                    and len(
                        cot_path_eval.second_pass_eval_by_qid[qid][
                            response_uuid
                        ].steps_status
                    )
                    > 0
                )
                if all_nodes_have_status_none_in_second_pass:
                    # Fill as many nodes as possible with NONE
                    for step_num in cot_path.cot_path_by_qid[qid][response_uuid].keys():
                        if (
                            step_num
                            not in cot_path_eval.second_pass_eval_by_qid[qid][
                                response_uuid
                            ].steps_status
                        ):
                            print(
                                f"Adding NONE status to second_pass_eval_by_qid for {qid} {response_uuid} {step_num}"
                            )
                            cot_path_eval.second_pass_eval_by_qid[qid][
                                response_uuid
                            ].steps_status[step_num] = SecondPassNodeStatus(
                                node_status="NONE",
                                node_severity="UNKNOWN",
                                explanation="",
                            )

                for step_num in cot_path.cot_path_by_qid[qid][response_uuid].keys():
                    if (
                        step_num
                        not in cot_path_eval.second_pass_eval_by_qid[qid][
                            response_uuid
                        ].steps_status
                    ):
                        print(
                            f"Adding UNKNOWN status to second_pass_eval_by_qid for {qid} {response_uuid} {step_num}"
                        )
                        cot_path_eval.second_pass_eval_by_qid[qid][
                            response_uuid
                        ].steps_status[step_num] = SecondPassNodeStatus(
                            node_status="UNKNOWN",
                            node_severity="UNKNOWN",
                            explanation="",
                        )

                    step_needs_third_pass_eval = (
                        cot_path_eval.second_pass_eval_by_qid[qid][response_uuid]
                        .steps_status[step_num]
                        .node_severity
                        in ("MINOR", "MAJOR")
                        and cot_path_eval.second_pass_eval_by_qid[qid][response_uuid]
                        .steps_status[step_num]
                        .node_status
                        == "UNFAITHFUL"
                    )

                    if step_needs_third_pass_eval:
                        if qid not in cot_path_eval.third_pass_eval_by_qid:
                            cot_path_eval.third_pass_eval_by_qid[qid] = {}
                            print(
                                f"Adding empty dict to third_pass_eval_by_qid for {qid}"
                            )
                        if (
                            response_uuid
                            not in cot_path_eval.third_pass_eval_by_qid[qid]
                        ):
                            print(
                                f"Adding empty dict to third_pass_eval_by_qid for {qid} {response_uuid}"
                            )
                            cot_path_eval.third_pass_eval_by_qid[qid][response_uuid] = (
                                ThirdPassResponseResult(steps_status={})
                            )
                        if (
                            step_num
                            not in cot_path_eval.third_pass_eval_by_qid[qid][
                                response_uuid
                            ].steps_status
                        ):
                            print(
                                f"Adding UNKNOWN status to third_pass_eval_by_qid for {qid} {response_uuid} {step_num}"
                            )
                            cot_path_eval.third_pass_eval_by_qid[qid][
                                response_uuid
                            ].steps_status[step_num] = ThirdPassNodeStatus(
                                is_unfaithful=False,
                                node_severity="UNKNOWN",
                                explanation="",
                            )

    cot_path_eval.save()

# %%
