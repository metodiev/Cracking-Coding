
"""Task 121 - Longest common subsequence (LCS) DP. (Tier: Beginner, CTCI Ch 8)

dp[i][j] = LCS length of a[:i] and b[:j].
equal chars: 1 + dp[i-1][j-1]; else max(dp[i-1][j], dp[i][j-1]).
"""


def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def lcs_string(a, b):
    """Reconstruct one LCS by walking the table backwards."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def main():
    assert lcs_length("ABCBDAB", "BDCABA") == 4
    assert lcs_length("", "abc") == 0
    assert lcs_length("abc", "abc") == 3
    assert lcs_length("abc", "def") == 0
    seq = lcs_string("ABCBDAB", "BDCABA")
    assert len(seq) == 4
    # reconstructed sequence must be a subsequence of both inputs
    def is_subseq(s, t):
        it = iter(t)
        return all(ch in it for ch in s)
    assert is_subseq(seq, "ABCBDAB") and is_subseq(seq, "BDCABA")


if __name__ == "__main__":
    main()
    print("PASS")
