from chainscope.cot_splitting import (
    check_steps_are_valid_split,
    parse_model_split_response,
)


def test_model_split_response_parsing():
    split_text = """Here's the reasoning with section markers:

<section 1>
1. **Atomic number** represents the number of protons in an atom's nucleus.
</section 1>

<section 2>
2. **Hydrogen** has 1 proton.
</section 2>

<section 3>
3. **Helium** has 2 protons.
</section 3>

<section 4>
Since hydrogen has fewer protons than helium, its atomic number is lower.
</section 4>

<section 5>
**Answer: NO** 
</section 5>"""
    sections = parse_model_split_response(split_text)

    assert sections == [
        "Here's the reasoning with section markers:",
        "1. **Atomic number** represents the number of protons in an atom's nucleus.",
        "2. **Hydrogen** has 1 proton.",
        "3. **Helium** has 2 protons.",
        "Since hydrogen has fewer protons than helium, its atomic number is lower.",
        "**Answer: NO**",
    ]


def test_check_steps_are_valid_split():
    original_response = """Here's the reasoning:

1. **Atomic number** represents the number of protons in an atom's nucleus.
2. **Hydrogen** has 1 proton.
3. **Helium** has 2 protons.

Since hydrogen has fewer protons than helium, its atomic number is lower.


**Answer: NO**"""

    sections = [
        "Here's the reasoning:",
        "1. **Atomic number** represents the number of protons in an atom's nucleus.",
        "2. **Hydrogen** has 1 proton.",
        "3. **Helium** has 2 protons.",
        "Since hydrogen has fewer protons than helium, its atomic number is lower.",
        "**Answer: NO**",
    ]
    assert check_steps_are_valid_split(original_response, sections)
