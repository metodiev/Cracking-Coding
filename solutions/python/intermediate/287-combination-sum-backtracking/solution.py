"""Task 287 - Combination sum with repetition via backtracking. (Tier: Intermediate, CTCI Ch 8)

All unique combinations of positive candidates that sum to the target, each candidate
reusable. The DFS walks candidates in increasing order and continues from the same
index after taking a value, so every multiset is generated exactly once; results are
sorted deterministically. Cross-checked against a per-candidate copy-counting oracle.
"""

def combination_sum(candidates, target):
    """Returns all unique combinations (each a sorted list) summing to target, sorted."""
    numbers = sorted(candidates)
    results = []
    current = []

    def backtrack(start, remaining):
        if remaining == 0:
            results.append(current[:])
            return
        for i in range(start, len(numbers)):
            if numbers[i] > remaining:
                break
            current.append(numbers[i])
            backtrack(i, remaining - numbers[i])   # same index: repetition allowed
            current.pop()

    backtrack(0, target)
    results.sort()
    return results


def brute_force_combinations(candidates, target):
    """Oracle: choose how many copies of each candidate to take (counts per candidate)."""
    numbers = sorted(candidates)
    results = []

    def build(index, remaining, chosen):
        if index == len(numbers):
            if remaining == 0:
                results.append(chosen)
            return
        value = numbers[index]
        copies = 0
        while value * copies <= remaining:
            build(index + 1, remaining - value * copies, chosen + [value] * copies)
            copies += 1

    build(0, target, [])
    return sorted(results)


class Lcg:
    """Deterministic generator: next = (next * 75 + 74) % 65537."""

    def __init__(self, seed=13579):
        self.state = seed

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def random_case(rng):
    size = 1 + rng.next() % 4
    candidates = []
    while len(candidates) < size:
        value = 1 + rng.next() % 8
        if value not in candidates:
            candidates.append(value)
    target = rng.next() % 13
    return candidates, target


def main():
    assert combination_sum([2, 3, 6, 7], 7) == [[2, 2, 3], [7]]
    assert combination_sum([2, 3, 5], 8) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
    assert combination_sum([7, 2, 3, 6], 7) == [[2, 2, 3], [7]]        # input order does not matter

    assert combination_sum([2, 3], 0) == [[]]                          # empty combination sums to 0
    assert combination_sum([2, 3], -1) == []                           # negative target has none
    assert combination_sum([2, 4], 7) == []                            # odd target, even values
    assert combination_sum([2], 1) == []                               # target below every candidate
    assert combination_sum([5], 10) == [[5, 5]]                        # repetition allowed
    assert combination_sum([8], 4) == []

    rng = Lcg()
    for trial in range(200):
        candidates, target = random_case(rng)
        results = combination_sum(candidates, target)
        assert results == brute_force_combinations(candidates, target), (candidates, target)
        seen = set()
        for combination in results:
            assert sum(combination) == target, (candidates, target, combination)
            assert all(value in candidates for value in combination), combination
            assert combination == sorted(combination), combination
            assert tuple(combination) not in seen, combination         # no duplicates
            seen.add(tuple(combination))
        assert len(results) == len(seen)


if __name__ == "__main__":
    main()
    print("PASS")
