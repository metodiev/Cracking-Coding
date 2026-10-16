"""Task 521 - Implement reservoir-sampling-based stratified sampling. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Stratified sampling where each stratum is drawn by reservoir sampling (Algorithm R). The
population is split into named strata; a total budget n is divided over the strata by the
largest-remainder (Hamilton) method on the exact fractional quotas n * size_i / total, with
ties broken by stratum index, so every stratum receives floor or ceil of its quota, the
allocations sum to exactly min(n, total), and an n == total draw is just the whole population.
Inside a stratum the sample of size k comes from a single pass with an O(k) reservoir: the first
k items fill the reservoir and each later item replaces slot rng() % (t+1). The generator is a
documented 64-bit LCG (seed * 6364136223846793005 + 1442695040888963407 mod 2^64) advanced once
per stream item, with draws taken from its top 31 bits so that a modulo-m draw stays essentially
unbiased - a naive mod-m of a small-modulus LCG is measurably skewed, which the
inclusion-frequency check below would detect. Documented checks: sampled sets are distinct
subsets of their stratum with the
allocated cardinality, the n == total case reproduces the population element-wise, allocation
matches floor/ceil of the quota with an exact sum, the same seed replays the same sample while
different seeds differ, k >= m and k = 1 edge cases, an empirical inclusion-frequency test that
every stream position of a 10-item stratum enters a 3-item reservoir in about 3/10 of 2000
trials for both Algorithm R and the independent Algorithm S (probability-proportional selection),
and a variance comparison showing that the stratified mean estimator beats the same-budget
simple random sample on a population whose variance is dominated by the between-stratum term.
"""


MASK = (1 << 64) - 1


def lcg(seed):
    """64-bit LCG step; the top 31 bits are near-uniform, see rand_below."""
    return (seed * 6364136223846793005 + 1442695040888963407) & MASK


def rand_below(seed, modulus):
    """Uniform-enough draw in [0, modulus) from the high bits of one LCG step."""
    return (seed >> 33) % modulus


def allocate(sizes, n):
    """Largest-remainder apportionment of n seats over the given stratum sizes."""
    total = sum(sizes)
    if total == 0:
        return [0] * len(sizes)
    n = min(n, total)
    quotas = [n * size / total for size in sizes]
    floor = [int(q) for q in quotas]
    remainder = n - sum(floor)
    order = sorted(range(len(sizes)), key=lambda i: (-(quotas[i] - floor[i]), i))
    for i in order[:remainder]:
        floor[i] += 1
    return floor


def reservoir_sample(stream, k, seed):
    """Algorithm R: single pass, O(k) memory; returns the reservoir in slot order."""
    reservoir = []
    for position, item in enumerate(stream):
        if position < k:
            reservoir.append(item)
        else:
            seed = lcg(seed)
            slot = rand_below(seed, position + 1)
            if slot < k:
                reservoir[slot] = item
    return reservoir


def selection_sample(stream, k, seed):
    """Algorithm S: independent method with the same uniform-k-subset distribution."""
    m = len(stream)
    if k >= m:
        return list(stream)
    out = []
    remaining = m
    for item in stream:
        if len(out) == k:
            break
        seed = lcg(seed)
        if rand_below(seed, remaining) < k - len(out):
            out.append(item)
        remaining -= 1
    return out


def stratified_sample(strata, n, seed):
    """strata: list of (name, population). Allocates n over the strata, reservoir per stratum."""
    sizes = [len(population) for _, population in strata]
    allocation = allocate(sizes, n)
    out = {}
    for (name, population), take in zip(strata, allocation):
        seed = lcg(seed)                             # distinct, deterministic stream per stratum
        if take >= len(population):
            out[name] = list(population)
        else:
            out[name] = reservoir_sample(population, take, seed)
    return out


def stratified_mean(strata, sample):
    """Post-stratified mean: weights are the population shares, not the sample shares."""
    total = sum(len(population) for _, population in strata)
    estimate = 0.0
    for name, population in strata:
        drawn = sample[name]
        if drawn:
            estimate += len(population) / total * (sum(drawn) / len(drawn))
    return estimate


def main():
    # Allocation: quotas 60/40 over 10 seats are 6/4 exactly; 3/1 over 10 seats rounds up the
    # smaller stratum; a 1-seat budget goes to the largest remainder.
    assert allocate([60, 40], 10) == [6, 4]
    assert allocate([30, 10], 10) == [8, 2]
    assert allocate([10, 10, 10], 4) == [2, 1, 1]
    assert allocate([100, 1], 10) == [10, 0]
    assert allocate([5, 5], 100) == [5, 5], "n larger than the population is clamped"
    assert allocate([0, 0], 5) == [0, 0]
    assert allocate([7, 3, 5], 0) == [0, 0, 0]
    for sizes, n in (([60, 40], 10), ([7, 3, 5], 4), ([13, 1, 1, 1], 9), ([5, 5, 5], 7)):
        allocation = allocate(sizes, n)
        total = sum(sizes)
        assert sum(allocation) == min(n, total), (sizes, n, allocation)
        for size, seats in zip(sizes, allocation):
            quota = min(n, total) * size / total
            assert quota - 1.0 <= seats <= quota + 1.0, "floor/ceil of the exact quota"
            assert seats <= size

    # Reservoir sampling basics: the sample is a distinct subset of the stream of size k.
    stream = ["id%02d" % i for i in range(10)]
    for k in (1, 3, 9, 10):
        sample = reservoir_sample(stream, k, 2024)
        assert len(sample) == k and len(set(sample)) == k
        assert set(sample) <= set(stream)
    assert reservoir_sample(stream, 10, 7) == stream, "k == m keeps the whole stream in order"
    assert reservoir_sample([], 3, 11) == []
    assert reservoir_sample(["only"], 3, 11) == ["only"], "fewer items than k"
    # Determinism: identical seed replays; a different seed changes the draw.
    assert reservoir_sample(stream, 3, 5) == reservoir_sample(stream, 3, 5)
    assert reservoir_sample(stream, 3, 5) != reservoir_sample(stream, 3, 6)

    # Empirical inclusion frequency: every position enters a 3-of-10 reservoir ~30% of the time,
    # for Algorithm R and for the independent Algorithm S.
    trials = 2000
    for sampler in (reservoir_sample, selection_sample):
        counts = [0] * 10
        for trial in range(trials):
            sample = sampler(stream, 3, 1000 + trial)
            assert len(sample) == 3 and len(set(sample)) == 3
            for item in sample:
                counts[int(item[-2:])] += 1
        for position, count in enumerate(counts):
            frequency = count / trials
            assert 0.26 <= frequency <= 0.34, (sampler.__name__, position, frequency)
        assert sum(counts) == 3 * trials, "every trial contributes exactly k items"

    # Stratified draw: cardinalities, subset property, determinism.
    strata = [("small", ["s%d" % i for i in range(30)]),
              ("big", ["b%d" % i for i in range(170)])]
    sample = stratified_sample(strata, 40, 2026)
    assert sorted(sample) == ["big", "small"]
    assert len(sample["small"]) == 6 and len(sample["big"]) == 34, sample
    assert set(sample["small"]) <= set(strata[0][1])
    assert set(sample["big"]) <= set(strata[1][1])
    assert stratified_sample(strata, 40, 2026) == sample, "same seed replays the same draw"
    assert stratified_sample(strata, 40, 2027) != sample
    full = stratified_sample(strata, 200, 2026)
    assert full["small"] == strata[0][1] and full["big"] == strata[1][1]

    # Variance reduction: population = 40 values near 0 and 160 values near 1000, so almost all
    # of the variance is between strata. A full census (n = 200) reproduces the population mean
    # exactly. At the same 100-item budget the stratified estimator (weights 0.2/0.8) is orders
    # of magnitude closer to the truth than a 100-item reservoir drawn from the pooled
    # population, whose sample mixture fluctuates hypergeometrically (measured: mean absolute
    # error 0.0147 and worst 0.057 stratified, versus 22.3 and 70.0 for the pooled draw).
    population = [(i % 100) / 100.0 for i in range(40)] + \
                 [1000.0 + (i % 100) / 100.0 for i in range(160)]
    strata = [("low", population[:40]), ("high", population[40:])]
    truth = sum(population) / len(population)
    census = stratified_sample(strata, 200, 5)
    assert abs(stratified_mean(strata, census) - truth) <= 1e-12, "census is exact"
    sample = stratified_sample(strata, 100, 5)
    assert len(sample["low"]) == 20 and len(sample["high"]) == 80, "proportional allocation"
    stratified_errors = []
    simple_errors = []
    for seed in range(1, 201):
        draw = stratified_sample(strata, 100, seed)
        stratified_errors.append(abs(stratified_mean(strata, draw) - truth))
        pooled = reservoir_sample(population, 100, seed)
        simple_errors.append(abs(sum(pooled) / len(pooled) - truth))
    mean_stratified = sum(stratified_errors) / len(stratified_errors)
    mean_simple = sum(simple_errors) / len(simple_errors)
    assert mean_stratified * 50.0 < mean_simple, (mean_stratified, mean_simple)
    assert max(stratified_errors) <= 0.1, max(stratified_errors)

    # A uniform population has no between-stratum variance, so the stratified estimator is still
    # unbiased there: over 120 seeds the average absolute error stays below 4 of a range of 200.
    flat = [float(i) for i in range(200)]
    flat_strata = [("a", flat[:100]), ("b", flat[100:])]
    flat_truth = sum(flat) / len(flat)
    flat_errors = [abs(stratified_mean(flat_strata, stratified_sample(flat_strata, 40, seed))
                       - flat_truth) for seed in range(1, 121)]
    assert sum(flat_errors) / len(flat_errors) <= 4.0, "unbiased on a uniform population"
    assert max(flat_errors) <= 12.0, max(flat_errors)


if __name__ == "__main__":
    main()
    print("PASS")
