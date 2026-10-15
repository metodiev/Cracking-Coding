"""Task 089 - Implement simple frequency histogram (integers). (Tier: Beginner, CTCI Ch 1)

histogram_counts(values) -> {value: count}, built in one pass.
render_histogram(counts) -> one text line per distinct value, sorted ascending,
rendered as "value: " followed by one "#" per occurrence, e.g. "3: ###".
"""


def histogram_counts(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def render_histogram(counts):
    return [f"{value}: {'#' * counts[value]}" for value in sorted(counts)]


def main():
    counts = histogram_counts([1, 2, 2, 3, 3, 3])
    assert counts == {1: 1, 2: 2, 3: 3}

    lines = render_histogram(counts)
    assert len(lines) == 3
    assert lines == ["1: #", "2: ##", "3: ###"]

    empty = histogram_counts([])
    assert empty == {}
    assert render_histogram(empty) == []

    negatives = histogram_counts([-1, -1, 0, 2, 2, 2])
    assert negatives == {-1: 2, 0: 1, 2: 3}
    assert render_histogram(negatives) == ["-1: ##", "0: #", "2: ###"]

    single = histogram_counts([7])
    assert single == {7: 1}
    assert render_histogram(single) == ["7: #"]


if __name__ == "__main__":
    main()
    print("PASS")
