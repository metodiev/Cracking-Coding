
"""Task 123 - Edit distance (Levenshtein) DP. (Tier: Beginner, CTCI Ch 8)

dp[i][j] = min(delete, insert, substitute); substitution is free when chars match.
"""


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i                # delete all of a
    for j in range(m + 1):
        dp[0][j] = j                # insert all of b
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # delete
                                   dp[i][j - 1],      # insert
                                   dp[i - 1][j - 1])  # substitute
    return dp[n][m]


def edit_distance_memory_optimized(a, b):
    """Same recurrence with two rolling rows: O(min) space."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = prev[j - 1] if ca == cb else 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[-1]


def main():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("same", "same") == 0
    assert edit_distance("flaw", "lawn") == 2
    # rolling-row version must agree
    for a, b in [("kitten", "sitting"), ("flaw", "lawn"), ("", ""), ("abc", "yabd")]:
        assert edit_distance(a, b) == edit_distance_memory_optimized(a, b), (a, b)


if __name__ == "__main__":
    main()
    print("PASS")
