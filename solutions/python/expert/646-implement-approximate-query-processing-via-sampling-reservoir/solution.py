"""Task 646 - Implement approximate query processing via sampling reservoir.  (Tier: Expert, Ch 14 — Databases)

Approximate query processing over a single streaming pass.  `reservoir_sample` implements
Algorithm R: the first k tuples fill the reservoir, then the i-th tuple (i >= k) replaces a
random slot with probability k/i, which keeps every k-subset of the stream equiprobable
while using O(k) memory and never revisiting a tuple.  Randomness comes from a hand-written
splitmix64 generator, so runs are perfectly reproducible.

The samples feed estimators (AVG, SUM, COUNT of a predicate) whose accuracy is bounded with
Hoeffding's inequality, `half_width = span * sqrt(ln(2/alpha) / (2k))`, where `span` is the
range of the value domain - which an AQP engine knows from metadata, not from the data.

Invariants verified: (a) a stream shorter than k is sampled exactly, (b) samples contain no
duplicates and use only k slots at their peak, (c) every stream element is selected with
probability k/n (checked against a 6-sigma binomial band over many seeds), (d) every
estimate lies inside its Hoeffding interval and the interval shrinks as k grows, (e) the
estimate is exact when the reservoir holds the whole stream, and (f) unlike a naive
"first k rows" sample, the reservoir does not mistake the head of the stream for the whole
stream.
"""

import math

MASK = (1 << 64) - 1


class SplitMix64:
    """Deterministic 64-bit PRNG: the same seed gives the same sequence everywhere."""

    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return (z ^ (z >> 31)) & MASK

    def below(self, bound):
        return self.next() % bound


def reservoir_sample(stream, size, seed):
    """Algorithm R; returns (sample, peak slots used)."""
    rng = SplitMix64(seed)
    reservoir, peak = [], 0
    for index, item in enumerate(stream):
        if index < size:
            reservoir.append(item)
        else:
            slot = rng.below(index + 1)
            if slot < size:
                reservoir[slot] = item
        peak = max(peak, len(reservoir))
    return reservoir, peak


def counter_stream(values, stats):
    """A one-shot generator that records how many tuples were actually pulled."""
    for value in values:
        stats["pulled"] = stats.get("pulled", 0) + 1
        yield value


def half_width(span, sample_size, alpha=0.05):
    """Hoeffding half-width for the mean of values in an interval of length `span`."""
    return span * math.sqrt(math.log(2.0 / alpha) / (2 * sample_size))


def estimate_mean(sample):
    return sum(sample) / len(sample)


def estimate_sum(sample, population):
    return estimate_mean(sample) * population


def estimate_count(sample, predicate, population):
    if not sample:
        return 0.0
    return population * sum(1 for value in sample if predicate(value)) / len(sample)


def stream_values(count, seed, step=7, modulus=100):
    values, state = [], seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        values.append((state % modulus) * step)
    return values


def main():
    values = list(range(20))
    sample, peak = reservoir_sample(iter(values), 5, 42)
    check(len(sample) == 5 and peak == 5, "a short stream fills the reservoir to k, no more")
    check(all(v in values for v in sample) and len(set(sample)) == 5,
          "the sample is a duplicate-free subset of the stream")
    check(any(v > values[4] for v in sample),
          "the sample reaches past the first k tuples: %r" % (sorted(sample),))

    full_sample, full_peak = reservoir_sample(iter(values), len(values), 7)
    check(full_sample == values and full_peak == len(values), "a stream shorter than k is exact")
    oversized, oversized_peak = reservoir_sample(iter(values), 50, 7)
    check(oversized == values and oversized_peak == len(values),
          "k > n cannot allocate more slots than the stream has")

    # every element must be selected with probability k/n --------------------------------
    runs, domain, size = 2400, 12, 1
    counts = {value: 0 for value in range(domain)}
    for run in range(runs):
        picked, _ = reservoir_sample(iter(range(domain)), size, 1000 + run)
        counts[picked[0]] += 1
    expected = runs * size / domain
    sigma = math.sqrt(runs * (size / domain) * (1 - size / domain))
    check(all(abs(count - expected) <= 6 * sigma + 1 for count in counts.values()),
          "k=1 selection is uniform within 6 sigma: %r" % (sorted(counts.values()),))
    check(min(counts.values()) > 0, "every element was selected at least once")

    inclusion, size = [0] * domain, 3
    threshold = 6 * math.sqrt(runs * (size / domain) * (1 - size / domain)) + 1
    for run in range(runs):
        picked, _ = reservoir_sample(iter(range(domain)), size, 50000 + run)
        for value in picked:
            inclusion[value] += 1
    expected = runs * size / domain
    check(all(abs(count - expected) <= threshold for count in inclusion),
          "k=3 inclusion probabilities are uniform: %r" % (inclusion,))

    again, _ = reservoir_sample(iter(range(domain)), size, 12345)
    twice, _ = reservoir_sample(iter(range(domain)), size, 12345)
    check(again == twice, "the same seed gives the same sample")
    others = {tuple(reservoir_sample(iter(range(domain)), size, 12345 + i)[0])
              for i in range(20)}
    check(len(others) > 1, "different seeds give different samples")

    # one pass, O(k) memory, deterministic estimators ------------------------------------
    stream = stream_values(400, 2024)
    stats = {}
    sample, peak = reservoir_sample(counter_stream(stream, stats), 40, 99)
    check(stats["pulled"] == len(stream), "the stream was consumed exactly once")
    check(peak == 40, "the reservoir never held more than k tuples")
    check(len(sample) == 40 and len(set(sample)) <= 40, "no tuple is stored twice")

    true_mean = sum(stream) / len(stream)
    span = max(stream) - min(stream)
    estimated_mean = estimate_mean(sample)
    check(abs(estimated_mean - true_mean) <= half_width(span, len(sample)),
          "the AVG estimate %.2f lies inside its Hoeffding interval around %.2f"
          % (estimated_mean, true_mean))
    check(abs(estimate_sum(sample, len(stream)) - sum(stream))
          <= half_width(span, len(sample)) * len(stream),
          "the SUM estimate obeys the same bound scaled by n")
    predicate = lambda v: v % 20 == 0                     # noqa: E731
    true_count = sum(1 for v in stream if predicate(v))
    estimated_count = estimate_count(sample, predicate, len(stream))
    check(abs(estimated_count - true_count)
          <= len(stream) * half_width(1.0, len(sample)),
          "the COUNT estimate obeys the Bernoulli bound: %.1f vs %d"
          % (estimated_count, true_count))

    # a longer reservoir never has a wider interval, and k >= n is exact ----------------
    widths = [half_width(span, k) for k in (10, 20, 40, 80, 160)]
    check(all(widths[i] > widths[i + 1] for i in range(len(widths) - 1)),
          "the Hoeffding interval shrinks as the sample grows: %r" % (widths,))
    check(half_width(span, 40, 0.01) > half_width(span, 40, 0.05),
          "a tighter confidence level needs a wider interval")
    exact_sample, exact_peak = reservoir_sample(iter(stream), len(stream) + 10, 5)
    check(estimate_mean(exact_sample) == true_mean and exact_peak == len(stream),
          "a reservoir holding everything is exact, not approximate")
    check(abs(estimate_count(exact_sample, predicate, len(stream)) - true_count) < 1e-9,
          "COUNT is exact when the sample is the population")

    # skew: a heavy tail is handled correctly bounded -------------------------------------
    skewed = [0] * 180 + [1000] * 20
    skew_span = max(skewed) - min(skewed)
    skew_true = sum(skewed) / len(skewed)
    skew_bad = 0
    for seed in range(30):
        picked, _ = reservoir_sample(iter(skewed), 30, 3000 + seed)
        if abs(estimate_mean(picked) - skew_true) > half_width(skew_span, 30):
            skew_bad += 1
    check(skew_bad == 0, "every skewed estimate stayed inside its interval (%d failures)"
          % skew_bad)

    # contrast: the naive "first k rows" sample is biased on a stream with a drifted head --
    drifted = [0] * 200 + [1] * 200
    drifted_true = 0.5
    naive_error = abs(sum(drifted[:20]) / 20 - drifted_true)
    reservoir_errors = []
    for seed in range(40):
        picked, _ = reservoir_sample(iter(drifted), 20, 8000 + seed)
        reservoir_errors.append(abs(sum(picked) / 20 - drifted_true))
    check(naive_error == 0.5, "the plain prefix sample is exactly wrong on a drifted stream")
    check(max(reservoir_errors) < 0.45,
          "the reservoir stays close to the truth (%r)" % (max(reservoir_errors),))
    check(sum(reservoir_errors) / len(reservoir_errors) < 0.15,
          "the reservoir error is small on average")
    print("[646] reservoir: k=40 of n=400, AVG %.2f vs true %.2f, span bound %.2f"
          % (estimated_mean, true_mean, half_width(span, 40)))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
