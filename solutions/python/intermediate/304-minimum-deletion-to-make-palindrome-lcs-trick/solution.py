"""Task 304 - Minimum deletions to make a palindrome (LCS trick). (Tier: Intermediate, CTCI Ch 1)

Keep a longest palindromic subsequence: deletions = len(s) - LPS(s), and
LPS(s) == LCS(s, reverse(s)). Both are computed independently here and must agree.
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


def longest_palindromic_subsequence(s):
    """Direct interval DP, independent of the LCS formulation."""
    n = len(s)
    if n == 0:
        return 0
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2 if length > 2 else 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]


def minimum_deletions(s):
    return len(s) - longest_palindromic_subsequence(s)


def lcg_strings(length, count, seed=3579):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("abcd"[seed % 4])
        strings.append("".join(chars))
    return strings


def main():
    assert minimum_deletions("aebcbda") == 2
    assert minimum_deletions("abc") == 2
    assert minimum_deletions("abcd") == 3
    assert minimum_deletions("aa") == 0
    assert minimum_deletions("racecar") == 0
    assert minimum_deletions("") == 0
    assert minimum_deletions("a") == 0

    for s in ["aebcbda", "geeksforgeeks", "abca", "abbc", ""]:
        assert longest_palindromic_subsequence(s) == lcs_length(s, s[::-1]), s

    for s in lcg_strings(10, 60):
        lps = longest_palindromic_subsequence(s)
        assert lps == lcs_length(s, s[::-1]), s
        assert 0 <= len(s) - lps <= len(s)


if __name__ == "__main__":
    main()
    print("PASS")
