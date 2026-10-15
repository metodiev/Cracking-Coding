"""Task 094 - Find longest common prefix in array of strings. (Tier: Beginner, CTCI Ch 8)

Vertical scan: walk the first string and compare the same position in every
other string; the first mismatch ends the prefix. Empty array returns "".
"""


def longest_common_prefix(strs):
    if not strs:
        return ""
    first = strs[0]
    for i, ch in enumerate(first):
        for s in strs[1:]:
            if i >= len(s) or s[i] != ch:
                return first[:i]
    return first


def main():
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix(["apple"]) == "apple"
    assert longest_common_prefix([]) == ""
    assert longest_common_prefix([""]) == ""
    assert longest_common_prefix(["", "a"]) == ""
    assert longest_common_prefix(["a", "ab"]) == "a"
    assert longest_common_prefix(["abc", "abc", "abc"]) == "abc"
    assert longest_common_prefix(["prefix", "pref"]) == "pref"
    assert longest_common_prefix(["ab", "ac"]) == "a"


if __name__ == "__main__":
    main()
    print("PASS")
