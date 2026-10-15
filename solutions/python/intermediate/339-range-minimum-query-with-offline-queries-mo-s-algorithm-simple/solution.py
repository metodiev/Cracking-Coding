"""Task 339 - Range minimum query with offline queries (Mo's algorithm simple). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Simplified Mo's algorithm: block-sort the offline queries by (l / B, r) with B = n / sqrt(q)
so that sweeping the window boundaries performs only O((n + q) * sqrt(n)) add/remove moves
(the classic Mo bound), each O(log n) amortised on a min-heap multiset with lazy deletion.
Answers every range minimum in query order; all answers are cross-checked against a naive
scan on deterministic LCG inputs and queries.
"""

import heapq


class LazyMinMultiset:
    """Multiset whose minimum comes from a min-heap with lazily deleted values."""

    def __init__(self):
        self.heap = []
        self.pending = {}
        self.size = 0

    def add(self, value):
        heapq.heappush(self.heap, value)
        self.size += 1

    def remove(self, value):
        self.pending[value] = self.pending.get(value, 0) + 1
        self.size -= 1

    def minimum(self):
        while self.heap and self.pending.get(self.heap[0], 0) > 0:
            value = heapq.heappop(self.heap)
            self.pending[value] -= 1
        assert self.size > 0, "minimum of an empty multiset"
        return self.heap[0]


def mo_range_min(values, queries):
    """Returns the minimum of values[left..right] for each query, in input order."""
    n = len(values)
    q = len(queries)
    if q == 0:
        return []
    root = int(q ** 0.5)
    block = max(1, n // root) if root > 0 else n
    order = sorted(
        range(q),
        key=lambda i: (queries[i][0] // block,
                       queries[i][1] if (queries[i][0] // block) % 2 == 0 else -queries[i][1]),
    )

    window = LazyMinMultiset()
    answers = [0] * q
    current_left, current_right = 0, -1
    for index in order:
        left, right = queries[index]
        while current_left > left:
            current_left -= 1
            window.add(values[current_left])
        while current_right < right:
            current_right += 1
            window.add(values[current_right])
        while current_left < left:
            window.remove(values[current_left])
            current_left += 1
        while current_right > right:
            window.remove(values[current_right])
            current_right -= 1
        answers[index] = window.minimum()
    return answers


def naive_range_min(values, queries):
    return [min(values[left:right + 1]) for left, right in queries]


def lcg_queries(count, seed, n):
    queries = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        left = seed % n
        seed = (seed * 75 + 74) % 65537
        right = left + seed % (n - left)
        queries.append((left, right))
    return queries, seed


def main():
    # Hand-checked minima on a small fixed array.
    values = [1, 2, 1, 3, 2, 4]
    queries = [(0, 2), (1, 4), (0, 5), (3, 3), (2, 5), (1, 1), (5, 5)]
    assert mo_range_min(values, queries) == [1, 1, 1, 3, 1, 2, 4]
    assert naive_range_min(values, queries) == [1, 1, 1, 3, 1, 2, 4]
    assert mo_range_min([7], [(0, 0)]) == [7]
    assert mo_range_min(values, []) == []

    # All-equal and strictly increasing arrays, including duplicate-heavy windows.
    flat = [5] * 10
    assert mo_range_min(flat, [(0, 9), (2, 3), (4, 4)]) == [5, 5, 5]
    rising = list(range(10))
    assert mo_range_min(rising, [(0, 9), (0, 0), (3, 7)]) == [0, 0, 3]
    twos = [2, 2, 3, 2, 3, 3]
    assert mo_range_min(twos, [(0, 5), (1, 4), (2, 5), (3, 5)]) == [2, 2, 2, 2]

    # LCG cross-check with a spread of sizes and query counts.
    seed = 5150
    for trial in range(12):
        seed = (seed * 75 + 74) % 65537
        n = 1 + seed % 120
        seed = (seed * 75 + 74) % 65537
        q = 1 + seed % 200
        seed = (seed * 75 + 74) % 65537
        alphabet = 1 + seed % 12
        values = []
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            values.append(seed % alphabet)
        queries, seed = lcg_queries(q, seed, n)
        fast = mo_range_min(values, queries)
        slow = naive_range_min(values, queries)
        assert fast == slow, (trial, n, q)
        for left, right in queries:
            assert 0 <= left <= right < n

    # Single-position queries, one full query, and an adversarially reversed query order.
    seed = 424242
    values = []
    for _ in range(50):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % 7)
    singles = [(i, i) for i in range(50)]
    assert mo_range_min(values, singles) == values
    assert mo_range_min(values, [(0, 49)]) == [min(values)]
    assert mo_range_min(values, singles[::-1]) == values[::-1]


if __name__ == "__main__":
    main()
    print("PASS")
