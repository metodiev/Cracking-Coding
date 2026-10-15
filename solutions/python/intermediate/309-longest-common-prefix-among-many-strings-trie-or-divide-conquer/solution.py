"""Task 309 - Longest common prefix among many strings. (Tier: Intermediate, CTCI Ch 1)

Only the lexicographic minimum and maximum of the list can bound the common prefix, so the
answer is instantly prefix(min, max). Validated against an independent vertical scan and a
trie-based implementation.
"""

def longest_common_prefix_sorted(strs):
    """min/max bound: the common prefix is the prefix of the two extremes."""
    if not strs:
        return ""
    smallest, largest = min(strs), max(strs)
    i = 0
    while i < len(smallest) and i < len(largest) and smallest[i] == largest[i]:
        i += 1
    return smallest[:i]


def longest_common_prefix_vertical(strs):
    """Column-by-column scan; the independent oracle."""
    if not strs:
        return ""
    for column in range(len(strs[0])):
        char = strs[0][column]
        for word in strs[1:]:
            if column >= len(word) or word[column] != char:
                return strs[0][:column]
    return strs[0]


def longest_common_prefix_trie(strs):
    """Trie walk: keep descending while a single child is present and no word ends here."""
    root = {}
    for word in strs:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node["$"] = True
    prefix = []
    node = root
    while len(node) == 1 and "$" not in node:
        char = next(iter(node))
        prefix.append(char)
        node = node[char]
    return "".join(prefix)


def lcg_lists(count, seed=1592):
    lists = []
    for _ in range(count):
        size = 1 + seed % 3
        words = []
        for _ in range(size):
            word = []
            length = 1 + seed % 4
            for _ in range(length):
                seed = (seed * 75 + 74) % 65537
                word.append("abc"[seed % 3])
            words.append("".join(word))
        seed = (seed * 75 + 74) % 65537
        lists.append(words)
    return lists


def main():
    assert longest_common_prefix_sorted(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix_sorted(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix_sorted(["abc", "abc"]) == "abc"
    assert longest_common_prefix_sorted(["a"]) == "a"
    assert longest_common_prefix_sorted([]) == ""
    assert longest_common_prefix_sorted(["", "b"]) == ""
    assert longest_common_prefix_sorted(["abab", "aba", "ababbb"]) == "aba"

    for strs in [["flower", "flow", "flight"], ["dog", "racecar", "car"], ["abc", "abc"], ["", "b"]]:
        expected = longest_common_prefix_sorted(strs)
        assert longest_common_prefix_vertical(strs) == expected, strs
        assert longest_common_prefix_trie(strs) == expected, strs

    for strs in lcg_lists(80):
        expected = longest_common_prefix_sorted(strs)
        assert longest_common_prefix_vertical(strs) == expected, strs
        assert longest_common_prefix_trie(strs) == expected, strs


if __name__ == "__main__":
    main()
    print("PASS")
