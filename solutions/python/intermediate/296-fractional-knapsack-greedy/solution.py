"""Task 296 - Fractional knapsack (greedy). (Tier: Intermediate, CTCI Ch 8)

Greedy by value/weight ratio: sort items by decreasing ratio, take whole items until the
capacity is exhausted, then a fraction of the next item. For the fractional relaxation
this is provably optimal (an exchange argument turns any optimum into the greedy one),
unlike the 0/1 case. Cross-checked against an exhaustive "subset of whole items plus one
fraction" oracle on deterministic LCG instances.
"""


def fractional_knapsack(weights, values, capacity):
    """Greedy by ratio; returns (total value, [(item index, fraction taken), ...])."""
    order = sorted(range(len(weights)), key=lambda i: (-values[i] / weights[i], i))
    remaining = float(capacity)
    total = 0.0
    picks = []
    for index in order:
        if remaining <= 0:
            break
        fraction = min(1.0, remaining / weights[index])
        total += fraction * values[index]
        remaining -= fraction * weights[index]
        picks.append((index, fraction))
    return total, picks


def brute_fractional(weights, values, capacity):
    """Oracle: any optimum takes whole items plus at most one fraction; enumerate both.

    The fraction is capped at 1 because the remaining item exists only once -- omitting
    the cap turns the oracle into the unbounded (fluid-supply) variant by mistake.
    """
    n = len(weights)
    best = 0.0
    for mask in range(1 << n):
        weight = 0
        value = 0
        for i in range(n):
            if (mask >> i) & 1:
                weight += weights[i]
                value += values[i]
        if weight > capacity:
            continue
        best = max(best, float(value))
        for j in range(n):
            if not (mask >> j) & 1:
                fraction = min(1.0, (capacity - weight) / weights[j])
                best = max(best, value + fraction * values[j])
    return best


def lcg_instance(n, seed):
    weights = []
    values = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        weights.append(1 + seed % 9)
        seed = (seed * 75 + 74) % 65537
        values.append(1 + seed % 9)
    return weights, values, seed


def main():
    total, picks = fractional_knapsack([10, 20, 30], [60, 100, 120], 50)
    assert abs(total - 240.0) < 1e-9, total
    assert [index for index, _ in picks] == [0, 1, 2]
    assert abs(picks[0][1] - 1.0) < 1e-9 and abs(picks[1][1] - 1.0) < 1e-9
    assert abs(picks[2][1] - 2 / 3) < 1e-9

    total, picks = fractional_knapsack([10, 20, 30], [60, 100, 120], 60)
    assert abs(total - 280.0) < 1e-9 and all(abs(f - 1.0) < 1e-9 for _, f in picks)

    total, picks = fractional_knapsack([10, 20, 30], [60, 100, 120], 0)
    assert total == 0.0 and picks == []

    total, picks = fractional_knapsack([5], [3], 2)
    assert abs(total - 1.2) < 1e-9 and abs(picks[0][1] - 0.4) < 1e-9

    seed = 424242
    for _ in range(120):
        seed = (seed * 75 + 74) % 65537
        n = 1 + seed % 8
        weights, values, seed = lcg_instance(n, seed)
        seed = (seed * 75 + 74) % 65537
        capacity = seed % 50
        total, picks = fractional_knapsack(weights, values, capacity)
        expected = brute_fractional(weights, values, capacity)
        assert abs(total - expected) < 1e-9, (weights, values, capacity, total, expected)
        assert sum(f * weights[i] for i, f in picks) <= capacity + 1e-9
        assert abs(sum(f * values[i] for i, f in picks) - total) < 1e-9
        assert all(0 < f <= 1.0 + 1e-12 for _, f in picks)


if __name__ == "__main__":
    main()
    print("PASS")
