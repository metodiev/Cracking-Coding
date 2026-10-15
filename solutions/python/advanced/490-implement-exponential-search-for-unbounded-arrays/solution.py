"""Task 490 - Implement exponential search for unbounded arrays. (Tier: Advanced, CTCI Ch 10)

Sorted ascending arrays are treated as conceptually unbounded: a documented sentinel sits at the
virtual index n (value +infinity), so the doubling phase may "look at" index n without ever reading
past n-1 - the peek() helper returns the sentinel for idx >= n and records the largest real index
touched. The search is cold-start: exponential_search returns the FIRST index holding target, or -1.

Algorithm (documented):
1. Peek index 0; if arr[0] >= target the range is [0, 0].
2. Otherwise double the bound b = 1, 2, 4, ... while b < n and arr[b] < target, remembering the last
   probed index prev with arr[prev] < target.
3. The first occurrence lies in [prev + 1, min(b, n - 1)]; finish with an ordinary lower-bound style
   binary search on that window and compare arr[lo] == target.

Probe budget: <= 1 + ceil(log2(i + 1)) doubling peeks plus <= ceil(log2(i)) window steps for a hit
at index i, bounded overall by 2 * ceil(log2(n + 1)) + 2 peeks (asserted). The answer is asserted
against both a plain binary search and a brute-force linear first-occurrence scan on LCG arrays
(with duplicates) including target at index 0, at n - 1, and absent values (sentinel case).
"""
import math


INF = float("inf")


def exponential_search(arr, target):
    """Returns (index, reads, max_real_index) where index is the first occurrence or -1."""
    n = len(arr)
    reads = 0
    max_index = -1

    def peek(idx):
        nonlocal reads, max_index
        reads += 1
        if idx >= n:
            return INF  # sentinel at the virtual index n; no out-of-range read happens
        if idx > max_index:
            max_index = idx
        return arr[idx]

    if n == 0:
        return -1, reads, max_index
    if peek(0) >= target:
        lo, hi = 0, 0
    else:
        bound = 1
        prev = 0
        while bound < n and peek(bound) < target:
            prev = bound
            bound <<= 1
        lo, hi = prev + 1, min(bound, n - 1)
    if lo > hi:
        return -1, reads, max_index
    while lo < hi:
        mid = (lo + hi) >> 1
        if peek(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    if peek(lo) == target:
        return lo, reads, max_index
    return -1, reads, max_index


def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        elif arr[mid] > target:
            hi = mid - 1
        else:
            while mid > 0 and arr[mid - 1] == target:
                mid -= 1
            return mid
    return -1


def brute_first(arr, target):
    for i, value in enumerate(arr):
        if value == target:
            return i
        if value > target:
            return -1
    return -1


def lcg_values(count, seed, limit):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % limit)
    return values


def main():
    # Hand cases.
    assert exponential_search([], 5)[0] == -1
    assert exponential_search([5], 5)[0] == 0
    assert exponential_search([5], 4)[0] == -1
    assert exponential_search([5], 6)[0] == -1
    assert exponential_search([1, 2, 3, 4, 5, 6, 7, 8], 1)[0] == 0
    assert exponential_search([1, 2, 3, 4, 5, 6, 7, 8], 8)[0] == 7
    assert exponential_search([1, 2, 3, 4, 5, 6, 7, 8], 4)[0] == 3
    assert exponential_search([1, 2, 3, 4, 5, 6, 7, 8], 0)[0] == -1
    assert exponential_search([1, 2, 3, 4, 5, 6, 7, 8], 9)[0] == -1
    assert exponential_search([2, 2, 2, 2], 2)[0] == 0
    assert exponential_search([2, 2, 2, 2], 3)[0] == -1
    assert exponential_search([0] * 8, 0)[0] == 0
    assert exponential_search([1, 3, 5, 7, 9, 11], 11)[0] == 5
    assert exponential_search([1, 3, 5, 7, 9, 11], 6)[0] == -1
    assert binary_search([1, 3, 5, 7, 9, 11], 6) == brute_first([1, 3, 5, 7, 9, 11], 6) == -1

    # LCG arrays with duplicates: exponential == binary == brute, bounds respected.
    found_total = 0
    absent_total = 0
    tests = 0
    max_reads = 0
    for seed in range(490, 495):
        arr = sorted(lcg_values(64, seed, 50))
        assert len(arr) == 64
        targets = list(arr)
        targets += [-1, 100, 49, 50]
        targets += [v % 60 for v in lcg_values(20, seed + 100, 60)]
        for target in targets:
            index, reads, max_index = exponential_search(arr, target)
            expected = brute_first(arr, target)
            assert index == expected, (seed, target, index, expected)
            assert index == binary_search(arr, target), (seed, target)
            assert max_index < len(arr), (seed, target, max_index)
            limit = 2 * math.ceil(math.log2(len(arr) + 1)) + 2
            assert reads <= limit, (seed, target, reads, limit)
            assert reads >= 1
            max_reads = max(max_reads, reads)
            tests += 1
            if index >= 0:
                found_total += 1
            else:
                absent_total += 1
    assert tests == 5 * 88, tests
    assert found_total == 389 and absent_total == 51, (found_total, absent_total)
    # Every arr[i] target is found; every absent probe stays absent.
    for seed in range(490, 495):
        arr = sorted(lcg_values(64, seed, 50))
        for i, value in enumerate(arr):
            index, reads, max_index = exponential_search(arr, value)
            assert index == brute_first(arr, value), (seed, i)
            assert max_index < 64
        for target in [-1, 100, 49, 50] + [v % 60 for v in lcg_values(20, seed + 100, 60)]:
            index, reads, max_index = exponential_search(arr, target)
            assert index == brute_first(arr, target)
            assert max_index < 64

    # Probe budget on a large array: the found index is never read out of order and reads stay in
    # the documented 2 * ceil(log2(n + 1)) + 2 envelope.
    n = 4096
    arr = sorted(lcg_values(n, 777, 100000))
    for i in range(0, n, 137):
        index, reads, max_index = exponential_search(arr, arr[i])
        assert index == brute_first(arr, arr[i])
        assert max_index < n
        assert reads <= 2 * math.ceil(math.log2(n + 1)) + 2
    assert exponential_search(arr, -5)[0] == -1
    assert exponential_search(arr, 100001)[0] == -1
    print("PASS")


if __name__ == "__main__":
    main()
