"""Task 308 - Count distinct subsequences. (Tier: Intermediate, CTCI Ch 1)

dp[i][j] = ways to build the first j characters of t from the first i characters of s;
if s[i] == t[j] the character may be used (dp[i-1][j-1]) or skipped (dp[i-1][j]).
Counts are exact Python integers; Java/JS/C# use big integers where needed.
"""

def count_distinct_subsequences(s, t):
    n, m = len(s), len(t)
    if m == 0:
        return 1
    if n < m:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[n][m]


def brute_force(s, t):
    """Enumerates every subsequence of s (fine for short strings)."""
    seen = set()
    for mask in range(1 << len(s)):
        candidate = "".join(s[i] for i in range(len(s)) if mask >> i & 1)
        if candidate == t:
            seen.add(mask)
    return len(seen)


def lcg_strings(length, count, seed=8642):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("ab"[seed % 2])
        strings.append("".join(chars))
    return strings


def main():
    assert count_distinct_subsequences("rabbbit", "rabbit") == 3
    assert count_distinct_subsequences("babgbag", "bag") == 5
    assert count_distinct_subsequences("abc", "") == 1
    assert count_distinct_subsequences("", "a") == 0
    assert count_distinct_subsequences("aaa", "a") == 3
    assert count_distinct_subsequences("aaa", "aa") == 3
    assert count_distinct_subsequences("abc", "abcd") == 0

    strings = lcg_strings(8, 40)
    for i in range(0, len(strings) - 1, 2):
        s, t = strings[i], strings[i + 1][:4]
        assert count_distinct_subsequences(s, t) == brute_force(s, t), (s, t)


if __name__ == "__main__":
    main()
    print("PASS")
