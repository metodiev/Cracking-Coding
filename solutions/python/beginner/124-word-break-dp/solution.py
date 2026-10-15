
"""Task 124 - Word break (DP). (Tier: Beginner, CTCI Ch 8)

dp[i] = can s[:i] be segmented into dictionary words.
dp[i] is true if some j < i has dp[j] true and s[j:i] in the dictionary.
O(n^2 * L) naive slicing; prefix checks can be sped up with a trie (task 116).
"""


def word_break(s, dictionary):
    words = set(dictionary)
    if not words:
        return s == ""
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[len(s)]


def word_break_segmentation(s, dictionary):
    """Return one valid segmentation as a list of words (or None)."""
    words = set(dictionary)
    dp = [False] * (len(s) + 1)
    prev = [-1] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                prev[i] = j
                break
    if not dp[len(s)]:
        return None
    parts = []
    i = len(s)
    while i > 0:
        j = prev[i]
        parts.append(s[j:i])
        i = j
    return parts[::-1]


def main():
    assert word_break("leetcode", ["leet", "code"])
    assert not word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
    assert word_break("applepenapple", ["apple", "pen"])
    assert word_break("", ["anything"]) or word_break("", [])
    assert not word_break("a", [])
    assert word_break_segmentation("leetcode", ["leet", "code"]) == ["leet", "code"]
    assert word_break_segmentation("catsandog", ["cats", "dog", "sand", "and", "cat"]) is None


if __name__ == "__main__":
    main()
    print("PASS")
