"""Task 310 - Suffix array by naively sorting suffixes. (Tier: Intermediate, CTCI Ch 1)

O(n^2 log n) baseline: sort the start indices by comparing the suffixes directly. The test
suite re-checks the ordering with a hand-written character-by-character comparator.
"""

def suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


def compare_suffixes(s, i, j):
    """Manual lexicographic comparator; the independent oracle."""
    n = len(s)
    while i < n and j < n:
        if s[i] != s[j]:
            return -1 if s[i] < s[j] else 1
        i += 1
        j += 1
    if i == j:
        return 0
    return -1 if i == n else 1          # the shorter suffix is smaller


def is_valid_suffix_array(s, order):
    if sorted(order) != list(range(len(s))):
        return False
    return all(compare_suffixes(s, order[k], order[k + 1]) < 0 for k in range(len(order) - 1))


def suffixes_in_order(s, order):
    return [s[i:] for i in order]


def lcg_strings(length, count, seed=2718):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("ab"[seed % 2])
        strings.append("".join(chars))
    return strings


def main():
    assert suffix_array("banana") == [5, 3, 1, 0, 4, 2]
    assert suffixes_in_order("banana", suffix_array("banana")) == ["a", "ana", "anana", "banana", "na", "nana"]

    mississippi = suffix_array("mississippi")
    assert mississippi == [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
    assert is_valid_suffix_array("mississippi", mississippi)

    assert suffix_array("") == []
    assert suffix_array("a") == [0]
    assert suffix_array("aaaa") == [3, 2, 1, 0]        # shorter suffixes sort first
    assert suffix_array("abab") == [2, 0, 3, 1]

    for s in ["banana", "mississippi", "aaaa", ""] + lcg_strings(12, 60):
        order = suffix_array(s)
        assert is_valid_suffix_array(s, order), s
        assert suffixes_in_order(s, order) == sorted(s[i:] for i in range(len(s))), s


if __name__ == "__main__":
    main()
    print("PASS")
