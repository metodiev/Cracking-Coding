r"""Task 200 - Build README template that documents test cases and complexity. (Tier: Beginner, CTCI Beyond)

render_section(name, tier, time_complexity, space_complexity, test_cases) renders the
Markdown section this repo uses to document every exercise: heading, tier line,
complexity table and test-case table. Table cells escape '|' as '\|' and newlines as
spaces; the returned string never has a trailing newline.
"""


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_section(name, tier, time_complexity, space_complexity, test_cases):
    lines = [
        "## %s" % name,
        "",
        "**Tier:** %s" % tier,
        "",
        "| Complexity | Value |",
        "| --- | --- |",
        "| Time | %s |" % _cell(time_complexity),
        "| Space | %s |" % _cell(space_complexity),
        "",
        "| Input | Expected |",
        "| --- | --- |",
    ]
    for input_value, expected in test_cases:
        lines.append("| `%s` | `%s` |" % (_cell(input_value), _cell(expected)))
    return "\n".join(lines).rstrip("\n")


def main():
    two_sum = render_section(
        "Two Sum", "Beginner", "O(n)", "O(n)",
        [("1, 2", "3"), ("[]", "0")],
    )
    assert two_sum == (
        "## Two Sum\n"
        "\n"
        "**Tier:** Beginner\n"
        "\n"
        "| Complexity | Value |\n"
        "| --- | --- |\n"
        "| Time | O(n) |\n"
        "| Space | O(n) |\n"
        "\n"
        "| Input | Expected |\n"
        "| --- | --- |\n"
        "| `1, 2` | `3` |\n"
        "| `[]` | `0` |"
    )

    escaped = render_section("Escape | Demo", "Beginner", "O(1)", "O(1)", [("a | b", "c\nd"), ("x\ny", "z")])
    assert escaped == (
        "## Escape | Demo\n"
        "\n"
        "**Tier:** Beginner\n"
        "\n"
        "| Complexity | Value |\n"
        "| --- | --- |\n"
        "| Time | O(1) |\n"
        "| Space | O(1) |\n"
        "\n"
        "| Input | Expected |\n"
        "| --- | --- |\n"
        "| `a \\| b` | `c d` |\n"
        "| `x y` | `z` |"
    )

    empty = render_section("Empty Cases", "Beginner", "O(1)", "O(1)", [])
    assert empty == (
        "## Empty Cases\n"
        "\n"
        "**Tier:** Beginner\n"
        "\n"
        "| Complexity | Value |\n"
        "| --- | --- |\n"
        "| Time | O(1) |\n"
        "| Space | O(1) |\n"
        "\n"
        "| Input | Expected |\n"
        "| --- | --- |"
    )
    assert not two_sum.endswith("\n") and not escaped.endswith("\n") and not empty.endswith("\n")


if __name__ == "__main__":
    main()
    print("PASS")
