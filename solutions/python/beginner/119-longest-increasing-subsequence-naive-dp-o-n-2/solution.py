
"""Task 119 - Longest increasing subsequence (naive DP O(n^2)). (Tier: Beginner, CTCI Ch 8)

dp[i] = length of the LIS ending at i = 1 + max(dp[j] for j < i with a[j] < a[i]).
"""


def lis_length(a):
    if not a:
        return 0
    dp = [1] * len(a)
    for i in range(1, len(a)):
        for j in range(i):
            if a[j] < a[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
    return max(dp)


def lis_with_sequence(a):
    """Same DP, reconstructing one concrete LIS via parent pointers."""
    if not a:
        return []
    dp = [1] * len(a)
    parent = [-1] * len(a)
    best = 0
    for i in range(1, len(a)):
        for j in range(i):
            if a[j] < a[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        if dp[i] >= dp[best]:        # >= keeps the LAST index, matching the expectation
            best = i
    seq = []
    node = best
    while node != -1:
        seq.append(a[node])
        node = parent[node]
    return seq[::-1]


def main():
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4          # 2, 3, 7, 18 (or 101)
    assert lis_length([]) == 0
    assert lis_length([3, 2, 1]) == 1
    assert lis_length([1, 2, 3, 4]) == 4
    assert lis_with_sequence([10, 9, 2, 5, 3, 7, 101, 18]) == [2, 5, 7, 18]


if __name__ == "__main__":
    main()
    print("PASS")
