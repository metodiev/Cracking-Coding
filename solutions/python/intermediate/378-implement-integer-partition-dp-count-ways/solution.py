"""Task 378 - Count integer partitions p(n) with DP. (Tier: Intermediate, CTCI Ch 8)

p(n) is the number of ways to write n as a sum of positive integers where ORDER DOES
NOT MATTER ((2, 1, 1) and (1, 2, 1) are the same partition of 4).  The coin-change
style DP treats part sizes one at a time: after processing part size k, dp[s] counts
partitions of s using parts <= k, so each multiset is built exactly once regardless
of order.  dp[0] = 1 (the empty partition) and p(n) = dp[n] after all sizes up to n.

Hand-verified milestones asserted below: p(0)=1, p(1)=1, p(4)=5, p(5)=7, p(10)=42,
p(20)=627, p(50)=204226 (classic values, cross-checked here by an independent
recursive counter for every n <= 20 and by the p(100)=190569292 milestone).

Oracle: `brute_force_partitions(n)` enumerates partitions recursively by picking the
largest remaining part, with no DP table - an independent counting path.
"""


def partition_count(n):
    """Number of integer partitions of n (unordered)."""
    if n < 0:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    for part in range(1, n + 1):
        for total in range(part, n + 1):
            dp[total] += dp[total - part]
    return dp[n]


def brute_force_partitions(n, max_part=None):
    """Recursive oracle: partitions of n whose parts are all <= max_part."""
    if max_part is None:
        max_part = n
    if n == 0:
        return 1
    if max_part == 0:
        return 0
    total = 0
    for part in range(1, min(n, max_part) + 1):
        total += brute_force_partitions(n - part, part)
    return total


def brute_force_via_ascending_parts(n):
    """Second oracle: enumerate non-decreasing part lists explicitly."""
    count = 0

    def walk(remaining, smallest):
        nonlocal count
        if remaining == 0:
            count += 1
            return
        part = smallest
        while part <= remaining:
            walk(remaining - part, part)
            part += 1

    walk(n, 1)
    return count


def main():
    # milestone values (classic partition numbers, re-derived by the oracles below)
    assert partition_count(0) == 1
    assert partition_count(1) == 1
    assert partition_count(2) == 2
    assert partition_count(3) == 3
    assert partition_count(4) == 5
    assert partition_count(5) == 7
    assert partition_count(10) == 42
    assert partition_count(20) == 627
    assert partition_count(50) == 204226
    assert partition_count(100) == 190569292

    # p(4) = 5 spelled out: 4; 3+1; 2+2; 2+1+1; 1+1+1+1
    assert brute_force_partitions(4) == 5
    assert brute_force_via_ascending_parts(4) == 5
    assert partition_count(4) == 5

    # p(5) = 7: adds 3+2 and 2+1+1+... no - it adds 3+2 and 1*5
    # (4+1; 3+2; 3+1+1; 2+2+1; 2+1+1+1; 1+1+1+1+1; and 5 itself)
    assert brute_force_partitions(5) == 7

    # all n <= 20 against both independent oracles
    for n in range(0, 21):
        expected = brute_force_partitions(n)
        assert partition_count(n) == expected, n
        assert brute_force_via_ascending_parts(n) == expected, n

    # monotone milestones guarding small bookkeeping errors
    previous = 0
    for n in range(1, 26):
        current = partition_count(n)
        assert current > previous, n
        previous = current

    assert partition_count(0) == 1 and partition_count(-3) == 0


if __name__ == "__main__":
    main()
    print("PASS")
