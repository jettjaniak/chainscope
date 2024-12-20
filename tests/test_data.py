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
        assert len(instructions.direct) > 0
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
