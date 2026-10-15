"""Task 337 - Design majority vote generalized (> n/k) using count map. (Tier: Intermediate, CTCI Ch 7)

Generalised Boyer-Moore vote: keep at most k - 1 candidate counters; a new value with no free
slot cancels one vote from every counter (drop zeros). Every element occurring more than n/k
times survives this cancellation, so a verification pass over the array filters the
survivors exactly. O(n * k) time, O(k) space. Cross-checked against a brute-force counting
answer on deterministic LCG arrays.
"""


def majority_elements(values, k):
    """Values with count * k > n (i.e. strictly more than n/k occurrences), sorted."""
    if k < 2:
        raise ValueError("k must be >= 2")
    n = len(values)
    if n == 0:
        return []

    slots = {}
    for value in values:
        if value in slots:
            slots[value] += 1
        elif len(slots) < k - 1:
            slots[value] = 1
        else:
            for candidate in list(slots):
                slots[candidate] -= 1
                if slots[candidate] == 0:
                    del slots[candidate]

    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(value for value in slots if counts[value] * k > n)


def brute_majority(values, k):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(value for value, count in counts.items() if count * k > len(values))


def lcg_values(count, seed, alphabet):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % alphabet)
    return values, seed


def main():
    # Hand-checked with n = 6, k = 4: the bar is count * 4 > 6, so counts 1 (3) and 2 (2)
    # qualify while 3 (count 1) does not.
    assert majority_elements([1, 1, 1, 2, 2, 3], 4) == [1, 2]
    assert brute_majority([1, 1, 1, 2, 2, 3], 4) == [1, 2]

    # Classic majority (k = 2): the bar is count * 2 > n.
    assert majority_elements([3, 2, 3], 2) == [3]
    assert majority_elements([2, 2, 1, 1, 1, 2, 2], 2) == [2]

    # No element passes the bar.
    assert majority_elements([1, 2, 3, 4], 4) == []
    assert majority_elements([1, 2, 3, 4], 2) == []
    assert majority_elements([1, 2, 2, 3, 3, 4], 3) == []
    assert majority_elements([], 3) == []

    # Exactly on the boundary: count * k == n is not "more than n/k".
    assert majority_elements([5, 5, 6, 6], 2) == []
    assert majority_elements([5, 5, 6, 6], 3) == [5, 6]

    # Every distinct value can pass when k is large enough.
    assert majority_elements([7, 8, 9, 7, 8, 9], 4) == [7, 8, 9]

    seed = 6060
    for trial in range(400):
        seed = (seed * 75 + 74) % 65537
        n = seed % 60
        seed = (seed * 75 + 74) % 65537
        k = 2 + seed % 5
        seed = (seed * 75 + 74) % 65537
        alphabet = 1 + seed % 9
        values, seed = lcg_values(n, seed, alphabet)
        found = majority_elements(values, k)
        expected = brute_majority(values, k)
        assert found == expected, (trial, k, values, found, expected)
        assert len(found) <= k - 1                      # at most k - 1 candidates survive
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        for value in found:
            assert counts[value] * k > n, (trial, value)   # every report really exceeds n/k


if __name__ == "__main__":
    main()
    print("PASS")
