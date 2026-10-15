
"""Task 165 - Reservoir sampling for k items from stream. (Tier: Beginner, CTCI Ch 6)

Algorithm R: keep the first k items; for the i-th item (i >= k), replace a random
slot with probability k/i. Every item ends up included with probability k/n.
A seeded RNG is injected so runs are reproducible and testable.
"""

import random


def reservoir_sample(stream, k, rng=None):
    rng = rng or random.Random()
    sample = []
    for i, item in enumerate(stream):
        if i < k:
            sample.append(item)
        else:
            j = rng.randrange(i + 1)
            if j < k:
                sample[j] = item
    return sample


def main():
    stream = list(range(100))
    sample = reservoir_sample(stream, 5, random.Random(7))
    assert len(sample) == 5
    assert len(set(sample)) == 5
    assert all(0 <= x < 100 for x in sample)
    # reproducibility with the same seed
    assert sample == reservoir_sample(stream, 5, random.Random(7))
    # different seed -> (almost surely) different sample
    assert sample != reservoir_sample(stream, 5, random.Random(8))

    # k >= n keeps everything
    assert reservoir_sample([1, 2, 3], 5, random.Random(1)) == [1, 2, 3]
    assert reservoir_sample([], 3, random.Random(1)) == []
    assert reservoir_sample([9], 1, random.Random(1)) == [9]

    # loose uniformity: element 0 should appear ~k/n = 5% of the time
    trials, hits = 4000, 0
    rng = random.Random(123)
    for _ in range(trials):
        if 0 in reservoir_sample(range(100), 5, rng):
            hits += 1
    ratio = hits / trials
    assert 0.03 < ratio < 0.08, f"inclusion ratio {ratio:.3f} out of expected ~0.05"


if __name__ == "__main__":
    main()
    print("PASS")
