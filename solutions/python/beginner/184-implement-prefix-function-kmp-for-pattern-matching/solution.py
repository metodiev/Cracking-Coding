"""Task 184 - Implement prefix function (KMP) for pattern matching. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)

pi[i] is the length of the longest proper prefix of s[:i+1] that is also a suffix
of s[:i+1] (the prefix's own prefix function gives the fallback position when
the current characters mismatch). Computed in O(n) with the classic fallback loop.
"""


def prefix_function(s):
    """Return the prefix function (LPS array) of `s`."""
    pi = [0] * len(s)
    for i in range(1, len(s)):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi


def prefix_function_naive(s):
    """O(n^2) reference: longest proper border of every prefix, brute-forced."""
    result = []
    for end in range(len(s)):
        best = 0
        for length in range(1, end + 1):
            if s[:length] == s[end - length + 1:end + 1]:
                best = length
        result.append(best)
    return result


def main():
    assert prefix_function("") == []
    assert prefix_function("a") == [0]
    assert prefix_function("ab") == [0, 0]
    assert prefix_function("ababaca") == [0, 0, 1, 2, 3, 0, 1]
    assert prefix_function("aabaaab") == [0, 1, 0, 1, 2, 2, 3]
    assert prefix_function("aaaa") == [0, 1, 2, 3]
    assert prefix_function("abcabcd") == [0, 0, 0, 1, 2, 3, 0]
    assert prefix_function("abababab") == [0, 0, 1, 2, 3, 4, 5, 6]
    assert prefix_function("aabaaabaaab") == [0, 1, 0, 1, 2, 2, 3, 4, 5, 6, 7]

    # agreement with the plain definition on pseudo-random strings
    import random
    rng = random.Random(5)
    for _ in range(200):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        assert prefix_function(s) == prefix_function_naive(s), s


if __name__ == "__main__":
    main()
    print("PASS")
