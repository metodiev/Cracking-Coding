"""Task 307 - Longest palindromic subsequence (DP). (Tier: Intermediate, CTCI Ch 1)

Interval DP: equal ends extend the inner answer by 2, otherwise drop one end.
Validated against an exhaustive subsequence enumeration on short strings.
"""

def longest_palindromic_subsequence(s):
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
                dp[i][j] = 2 if length == 2 else dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]


def brute_force(s):
    best = 0
    for mask in range(1 << len(s)):
        candidate = "".join(s[i] for i in range(len(s)) if mask >> i & 1)
        if len(candidate) > best and candidate == candidate[::-1]:
            best = len(candidate)
    return best


def lcg_strings(length, count, seed=9753):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("abc"[seed % 3])
        strings.append("".join(chars))
    return strings


def main():
    assert longest_palindromic_subsequence("bbbab") == 4
    assert longest_palindromic_subsequence("cbbd") == 2
    assert longest_palindromic_subsequence("abcde") == 1
    assert longest_palindromic_subsequence("racecar") == 7
    assert longest_palindromic_subsequence("") == 0
    assert longest_palindromic_subsequence("a") == 1

    for s in lcg_strings(10, 60):
        assert longest_palindromic_subsequence(s) == brute_force(s), s


if __name__ == "__main__":
    main()
    print("PASS")
