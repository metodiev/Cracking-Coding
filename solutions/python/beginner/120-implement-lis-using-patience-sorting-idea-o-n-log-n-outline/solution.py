
"""Task 120 - Implement LIS using patience sorting idea (O(n log n) outline). (Tier: Beginner, CTCI Ch 8)

Maintain `tails`: tails[i] = smallest possible tail of an increasing subsequence of
length i+1. Binary search find place to overwrite/append. Length is exact; the
`tails` array itself is NOT necessarily a valid subsequence (documented caveat).
"""

import bisect


def lis_length(a):
    tails = []
    for x in a:
        i = bisect.bisect_left(tails, x)      # strict increase: use bisect_left
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def lis_length_reconstructed(a):
    """O(n log n) length plus one valid LIS via parent tracking."""
    if not a:
        return 0, []
    tails = []           # values
    tail_idx = []        # index in a of each tail
    parent = [-1] * len(a)
    for i, x in enumerate(a):
        j = bisect.bisect_left(tails, x)
        if j == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[j] = x
            tail_idx[j] = i
        parent[i] = tail_idx[j - 1] if j > 0 else -1
    length = len(tails)
    seq = []
    node = tail_idx[-1]
    while node != -1:
        seq.append(a[node])
        node = parent[node]
    return length, seq[::-1]


def main():
    a = [10, 9, 2, 5, 3, 7, 101, 18]
    assert lis_length(a) == 4
    assert lis_length([]) == 0
    assert lis_length([3, 2, 1]) == 1
    assert lis_length([1, 2, 3, 4]) == 4
    length, seq = lis_length_reconstructed(a)
    assert length == 4
    assert seq == sorted(seq) and len(seq) == 4          # valid increasing subsequence

    # agrees with the naive DP (task 119) on many shapes
    cases = [[1, 3, 2], [5, 1, 2, 3, 4], [2, 2, 2], [-1, -2, -3]]
    for c in cases:
        naive = 0
        dp = [1] * len(c)
        for i in range(len(c)):
            for j in range(i):
                if c[j] < c[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
            naive = max(naive, dp[i]) if dp else 0
        assert lis_length(c) == naive, c


if __name__ == "__main__":
    main()
    print("PASS")
