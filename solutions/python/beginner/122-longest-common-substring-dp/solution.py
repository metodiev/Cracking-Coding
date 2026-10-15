
"""Task 122 - Longest common substring (DP). (Tier: Beginner, CTCI Ch 8)

Unlike LCS, substrings are contiguous:
dp[i][j] = dp[i-1][j-1] + 1 when a[i-1] == b[j-1], else 0. Track the max.
"""


def longest_common_substring(a, b):
    """Returns (length, substring)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best_len = 0
    best_end = 0            # index in a where the best run ends (exclusive)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_end = i
    return best_len, a[best_end - best_len:best_end]


def main():
    assert longest_common_substring("abcdef", "zbcdf") == (3, "bcd")
    assert longest_common_substring("abc", "abc") == (3, "abc")
    assert longest_common_substring("abc", "xyz") == (0, "")
    assert longest_common_substring("", "abc") == (0, "")
    # non-contiguous chars must NOT count: only single chars are common here.
    # (ties resolve to the first row-major match -> "a")
    assert longest_common_substring("axbxc", "abc") == (1, "a")


if __name__ == "__main__":
    main()
    print("PASS")
