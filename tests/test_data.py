from chainscope import DATA_DIR
from chainscope.typing import *


def test_load_instructions():
    """Test that all instruction sets in the YAML file can be loaded."""
    # Load raw YAML to get all instruction IDs
    with open(DATA_DIR / "instructions.yaml", "r") as f:
        raw_data = yaml.safe_load(f)

    # Try loading each instruction set
    for instr_id in raw_data.keys():
        instructions = Instructions.load(instr_id)
        assert isinstance(instructions, Instructions)
        assert isinstance(instructions.cot, str)
        assert isinstance(instructions.direct, str)
        assert len(instructions.cot) > 0
        for instruction in [instructions.cot, instructions.direct]:
            for line in instruction.splitlines():
                assert not line.startswith(" ")


def test_load_properties():
    """Test that all property files in data/properties can be loaded."""
    property_dir = DATA_DIR / "properties"

    # Get all YAML files in the properties directory
    property_files = list(property_dir.glob("*.yaml"))
    assert len(property_files) > 0, "No property files found"

    for prop_file in property_files:
        prop_id = prop_file.stem  # filename without extension
        properties = Properties.load(prop_id)

        # Check that it loaded correctly
        assert isinstance(properties, Properties)
        assert isinstance(properties.gt_question, str)
        assert isinstance(properties.lt_question, str)
        assert isinstance(properties.value_by_name, dict)
        assert len(properties.value_by_name) > 0

        for question in [properties.gt_question, properties.lt_question]:
            # Check that questions contain placeholders
            assert "{x}" in question
            assert "{y}" in question


def test_load_questions():
    """Test that question datasets can be loaded."""
    questions_dir = DATA_DIR / "questions"
    assert questions_dir.exists(), "Questions directory not found"

    # Get all YAML files in the questions directory and subdirectories
    question_files = list(questions_dir.rglob("*.yaml"))
    assert len(question_files) > 0, "No question files found"

    for question_file in question_files:
        QsDataset.load(question_file.stem)


def test_load_cot_responses():
    """Test that chain-of-thought responses can be loaded."""
    cot_dir = DATA_DIR / "cot_responses" / "instr-wm"
    assert cot_dir.exists(), "CoT responses directory not found"

    # Get all YAML files in the cot_responses directory and subdirectories
    cot_files = list(cot_dir.rglob("*.yaml"))
    assert len(cot_files) > 0, "No CoT response files found"

    for cot_file in cot_files:
        CotResponses.load(cot_file)


def test_load_cot_eval():
    """Test that chain-of-thought evaluations can be loaded."""
    eval_dir = DATA_DIR / "cot_eval" / "instr-wm"
    assert eval_dir.exists(), "CoT eval directory not found"

    # Get all YAML files in the cot_eval directory and subdirectories
    eval_files = list(eval_dir.rglob("*.yaml"))
    assert len(eval_files) > 0, "No CoT eval files found"

    for eval_file in eval_files:
        CotEval.from_yaml_file(eval_file)


def test_load_direct_eval():
    """Test that direct evaluations can be loaded."""
    eval_dir = DATA_DIR / "direct_eval"
    assert eval_dir.exists(), "Direct eval directory not found"

    # Get all YAML files in the direct_eval directory and subdirectories
    eval_files = list(eval_dir.rglob("*.yaml"))
    assert len(eval_files) > 0, "No direct eval files found"

    for eval_file in eval_files:
        DirectEval.from_yaml_file(eval_file)
