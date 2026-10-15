"""Task 305 - Shortest common supersequence length (DP). (Tier: Intermediate, CTCI Ch 1)

SCS(a, b) = |a| + |b| - LCS(a, b): every shared character is written once instead of twice.
Both the direct DP and the LCS formula are computed and asserted to agree.
"""

def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[len(b)]


def scs_length(a, b):
    """Direct DP: shortest string containing both a and b as subsequences."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[n][m]


def is_subsequence(candidate, text):
    it = iter(text)
    return all(ch in it for ch in candidate)


def lcg_strings(length, count, seed=4682):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("abc"[seed % 3])
        strings.append("".join(chars))
    return strings


def main():
    assert scs_length("AGGTAB", "GXTXAYB") == 9
    assert scs_length("ab", "b") == 2
    assert scs_length("", "abc") == 3
    assert scs_length("abc", "") == 3
    assert scs_length("", "") == 0
    assert scs_length("abc", "abc") == 3
    assert scs_length("geek", "eke") == 5       # "geeke"

    pairs = [("AGGTAB", "GXTXAYB"), ("geek", "eke"), ("ab", "b")]
    for a, b in pairs:
        assert scs_length(a, b) == len(a) + len(b) - lcs_length(a, b), (a, b)

    strings = lcg_strings(6, 40)
    for i in range(0, len(strings) - 1, 2):
        a, b = strings[i], strings[i + 1]
        assert scs_length(a, b) == len(a) + len(b) - lcs_length(a, b), (a, b)
        assert scs_length(a, b) >= max(len(a), len(b))


if __name__ == "__main__":
    main()
    print("PASS")
