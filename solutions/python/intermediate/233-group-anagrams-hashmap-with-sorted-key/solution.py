"""Task 233 - Group anagrams (hashmap with sorted key). (Tier: Intermediate, CTCI Ch 10)

Key each word by its characters sorted ("eat"/"tea"/"ate" -> "aet") and collect the
words in a dict keyed by that signature: O(n * L log L) for n words of length L.

Deterministic order: groups appear in the order their key is first seen in the input,
and words keep their input order inside a group.
"""


def group_anagrams(words):
    groups = {}
    for word in words:
        groups.setdefault("".join(sorted(word)), []).append(word)
    return list(groups.values())


def main():
    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == \
        [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    assert group_anagrams([]) == []
    assert group_anagrams([""]) == [[""]]
    assert group_anagrams(["", ""]) == [["", ""]]
    assert group_anagrams(["a"]) == [["a"]]
    assert group_anagrams(["ab", "ba", "abc", "cab", "bca"]) == \
        [["ab", "ba"], ["abc", "cab", "bca"]]
    assert group_anagrams(["a", "b", "a"]) == [["a", "a"], ["b"]]
    assert group_anagrams(["abc", "abc"]) == [["abc", "abc"]]
    assert group_anagrams(["listen", "silent", "enlist", "google"]) == \
        [["listen", "silent", "enlist"], ["google"]]


if __name__ == "__main__":
    main()
    print("PASS")
