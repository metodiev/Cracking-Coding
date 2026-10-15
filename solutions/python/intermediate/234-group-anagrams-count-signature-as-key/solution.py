"""Task 234 - Group anagrams (count signature as key). (Tier: Intermediate, CTCI Ch 1)

Same grouping rule as task 233, but the key is a 26-entry count signature (how many
'a's, 'b's, ... the word contains) instead of the sorted word: O(n * L) time, no
sorting. Inputs are assumed to be lowercase (the signature models only 'a'..'z').

Deterministic order (identical to 233): groups appear in first-seen key order and
words keep their input order inside a group, so both approaches return equal output;
the tests assert that agreement on several inputs.
"""


def group_anagrams_signature(words):
    groups = {}
    for word in words:
        counts = [0] * 26
        for ch in word:
            counts[ord(ch) - ord("a")] += 1
        groups.setdefault(tuple(counts), []).append(word)
    return list(groups.values())


def group_anagrams_sorted(words):
    """Task 233's sorted-key variant, kept here only for the agreement checks."""
    groups = {}
    for word in words:
        groups.setdefault("".join(sorted(word)), []).append(word)
    return list(groups.values())


def main():
    cases = [
        ["eat", "tea", "tan", "ate", "nat", "bat"],
        ["listen", "silent", "enlist", "google", "banana", "banana"],
        ["", "a", "", "a", "ab", "ba"],
        ["abc", "abc"],
        ["z"],
        [],
    ]
    for words in cases:                           # both approaches must agree
        assert group_anagrams_signature(words) == group_anagrams_sorted(words)

    assert group_anagrams_signature(["eat", "tea", "tan", "ate", "nat", "bat"]) == \
        [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    assert group_anagrams_signature([]) == []
    assert group_anagrams_signature([""]) == [[""]]
    assert group_anagrams_signature(["a", "b", "a"]) == [["a", "a"], ["b"]]
    assert group_anagrams_signature(["ab", "ba", "abc", "cab", "bca"]) == \
        [["ab", "ba"], ["abc", "cab", "bca"]]

    import random

    rng = random.Random(234)
    for _ in range(200):                          # random agreement check
        words = ["".join(rng.choice("abc") for _ in range(rng.randint(0, 6)))
                 for _ in range(rng.randint(0, 10))]
        assert group_anagrams_signature(words) == group_anagrams_sorted(words)


if __name__ == "__main__":
    main()
    print("PASS")
