"""Task 901 - Implement provable bounds for randomized algorithms used in your projects, with tests.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; the single randomness source is the seeded
splitmix64 stream in `Rng`, so every bound below is re-derivable on any platform):

  * `randomized_quicksort` - three-way (Dutch-national-flag) partitioning around a uniformly
    random pivot.  One unit is charged per non-pivot element per partition, i.e. a partition of
    a size-m subarray costs exactly m-1 comparisons, which is the counting convention of the
    classic analysis.  Under it the indicator identity holds:
        C_n = sum_{i<j} X_ij,   E[X_ij] = 2/(j-i+1),   E[C_n] = sum_{i<j} 2/(j-i+1) = 2(n+1)H_n - 4n,
    so the closed form and the indicator sum are two independent derivations of the mean.
  * `randomized_select` - quickselect with the same partition and one-sided recursion; the
    expected subproblem size shrinks by a constant factor per level, hence E[C] = O(n).
  * `chernoff_upper`, `hoeffding_two_sided` - the closed-form concentration inequalities, with
    `binom_tail_at_least` as an *exact rational* oracle (Fractions) for the true tail, so each
    bound is compared with ground truth rather than with itself.
  * majority (median) amplification of a 2/3-correct subroutine, with its Hoeffding bound.

Simplified versus production: the "randomized" subroutine is a deterministic PRNG-driven
quicksort, the tail-probability checks are exhaustive in a small model (binomial draws) rather
than over an unbounded family, and the high-probability comparison bound is validated by a
seeded seed sweep instead of the pairwise-independence induction that proves it.

Verified below: agreement with the `sorted()` oracle, the empty/single/all-equal boundaries,
stream determinism, exact agreement of the two expectation derivations, the empirical mean
against the analytic mean, the 4n lg n high-probability comparison bound, a negative control
(fabricated tighter bounds must be violated), quickselect's linear bound, Chernoff and
Hoeffding against the exact tail, and the amplification bound.
"""

import math
from fractions import Fraction

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64: the only randomness in this file, fully reproducible from its seed."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n):
        return self.next_u64() % n

    def bernoulli(self, numerator, denominator):
        """Exact Bernoulli(numerator/denominator) draw - no floating point involved."""
        return self.next_u64() % denominator < numerator


def partition_three_way(items, lo, hi, pivot_index, stats):
    """DNF partition of items[lo..hi]; the pivot is stashed at `lo` and every other element is
    compared with it exactly once.  Returns (lt, gt) with items[lo..lt-1] < pivot == items[lt..gt]
    < items[gt+1..hi]."""
    pivot = items[pivot_index]
    items[lo], items[pivot_index] = items[pivot_index], items[lo]
    lt, gt, i = lo, hi, lo + 1
    while i <= gt:
        stats[0] += 1
        if items[i] < pivot:
            items[lt], items[i] = items[i], items[lt]
            lt += 1
            i += 1
        elif items[i] > pivot:
            items[i], items[gt] = items[gt], items[i]
            gt -= 1
        else:
            i += 1
    return lt, gt


def randomized_quicksort(values, rng):
    """Three-way quicksort with a uniform random pivot.  Returns (sorted copy, comparisons)."""
    items = list(values)
    stats = [0]

    def sort(lo, hi):
        while lo < hi:
            lt, gt = partition_three_way(items, lo, hi, lo + rng.below(hi - lo + 1), stats)
            if lt - lo < hi - gt:            # recurse into the smaller side, loop on the larger
                sort(lo, lt - 1)
                lo = gt + 1
            else:
                sort(gt + 1, hi)
                hi = lt - 1

    sort(0, len(items) - 1)
    return items, stats[0]


def randomized_select(values, k, rng):
    """k-th smallest (0-based) by quickselect; returns (value, comparisons)."""
    items = list(values)
    stats = [0]
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        lt, gt = partition_three_way(items, lo, hi, lo + rng.below(hi - lo + 1), stats)
        if k < lt:
            hi = lt - 1
        elif k > gt:
            lo = gt + 1
        else:
            return items[lt], stats[0]
    raise AssertionError("select: k=%d out of range for n=%d" % (k, len(values)))


def harmonic(n):
    """H_n as an exact Fraction."""
    return sum((Fraction(1, i) for i in range(1, n + 1)), Fraction(0))


def expectation_closed_form(n):
    """E[C_n] = 2(n+1)H_n - 4n - the textbook closed form for randomized quicksort."""
    return 2 * (n + 1) * harmonic(n) - 4 * n if n else Fraction(0)


def expectation_indicator_sum(n):
    """E[C_n] = sum_{1<=i<j<=n} 2/(j-i+1) - the derivation from the comparison indicators."""
    total = Fraction(0)
    for j in range(1, n + 1):
        for i in range(1, j):
            total += Fraction(2, j - i + 1)
    return total


def permutation(n, rng):
    items = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        items[i], items[j] = items[j], items[i]
    return items


def chernoff_upper(mu, delta):
    """P[X >= (1+delta) mu] <= exp(-mu delta^2 / 3) for delta in (0, 1]."""
    return math.exp(-mu * delta * delta / 3.0)


def hoeffding_two_sided(t, n):
    """P[|S/n - E| >= t] <= 2 exp(-2 n t^2) for independent bounded summands in [0, 1]."""
    return 2.0 * math.exp(-2.0 * n * t * t)


def binom_tail_at_least(n, p, k):
    """Exact P[X >= k] for X ~ Binomial(n, p) using rational arithmetic (the oracle)."""
    if k <= 0:
        return Fraction(1)
    return sum((Fraction(math.comb(n, i)) * p ** i * (1 - p) ** (n - i)
                for i in range(k, n + 1)), Fraction(0))


def main():
    # ---- 1. correctness against an independent oracle and the degenerate boundaries
    for n in range(0, 49):
        rng = Rng(900 + n)
        values = [rng.below(37) for _ in range(n)]
        out, comps = randomized_quicksort(values, Rng(11))
        check(out == sorted(values), "quicksort must match sorted() for n=%d" % n)
        check(len(out) == n, "quicksort must not lose elements (n=%d)" % n)
        lower = 0 if n < 2 else n - 1     # one partition of m elements costs exactly m-1
        check(comps >= lower, "comparison count below the trivial bound (n=%d)" % n)

    flat = [7] * 40
    out, comps = randomized_quicksort(flat, Rng(3))
    check(out == flat and comps == len(flat) - 1,
          "an all-equal input must be partitioned once, i.e. n-1 comparisons")

    # ---- 2. determinism of the seeded stream
    values = permutation(64, Rng(21))
    first = randomized_quicksort(values, Rng(5))
    second = randomized_quicksort(values, Rng(5))
    check(first == second, "the same seed must replay the same run, comparisons included")
    counts = {randomized_quicksort(values, Rng(s))[1] for s in range(24)}
    check(len(counts) > 1, "different seeds must produce different runtimes")
    check(all(randomized_quicksort(values, Rng(s))[0] == sorted(values) for s in range(24)),
          "every seed must produce a correct result")

    # ---- 3. the provable mean: two independent derivations must agree exactly
    for n in range(1, 65):
        check(expectation_indicator_sum(n) == expectation_closed_form(n),
              "closed form and indicator sum disagree at n=%d" % n)
    check(expectation_closed_form(2) == 1 and expectation_closed_form(3) == Fraction(8, 3),
          "boundary values of the analytic mean")

    # ---- 4. the empirical mean tracks the analytic mean (n log n, not n^2)
    for n in (16, 32):
        runs = [randomized_quicksort(permutation(n, Rng(1000 + t)), Rng(t))[1] for t in range(300)]
        mean = Fraction(sum(runs), len(runs))
        analytic = expectation_closed_form(n)
        check(abs(mean - analytic) <= analytic / 10,
              "empirical mean %s vs analytic %s at n=%d" % (float(mean), float(analytic), n))
        high_prob = 4 * n * math.ceil(math.log2(n))
        check(max(runs) <= high_prob,
              "the 4 n lg n high-probability bound must hold (n=%d, max=%d)" % (n, max(runs)))
        # negative control: the tight fabricated bound 2n is genuinely violated
        check(sum(1 for c in runs if c > 2 * n) > len(runs) // 10,
              "the check must have teeth: runs above 2n must be observed (n=%d)" % n)

    # ---- 5. quickselect: correct and linear in expectation
    base = permutation(17, Rng(77))
    for k in range(17):
        got, _ = randomized_select(base, k, Rng(4))
        check(got == sorted(base)[k], "quickselect must return the k-th smallest (k=%d)" % k)
    sizes = 200
    sel_runs = [randomized_select(permutation(sizes, Rng(500 + t)), sizes // 2, Rng(t))[1]
                for t in range(100)]
    check(max(sel_runs) <= 16 * sizes, "quickselect must stay linear: max=%d n=%d" % (max(sel_runs), sizes))
    check(Fraction(sum(sel_runs), len(sel_runs)) <= 6 * sizes,
          "quickselect mean comparisons must be O(n)")

    # ---- 6. Chernoff and Hoeffding against the exact binomial oracle
    for (n, p, delta) in ((40, Fraction(1, 2), Fraction(1, 2)), (100, Fraction(1, 4), Fraction(1, 4)),
                          (60, Fraction(3, 5), Fraction(1, 5)), (25, Fraction(1, 10), Fraction(1, 2))):
        mu = n * p
        k = int(-(-mu * (1 + delta) // 1))
        exact = binom_tail_at_least(n, p, max(k, 0))
        bound = Fraction(chernoff_upper(float(mu), float(delta))).limit_denominator(10 ** 12)
        check(bound >= exact, "Chernoff must upper-bound the exact tail (n=%d)" % n)

        t = float(delta) * float(p)
        k_hi = int(math.ceil(float(n) * (float(p) + t)))
        k_lo = int(math.floor(float(n) * (float(p) - t)))
        exact_two = binom_tail_at_least(n, p, k_hi) + 1 - binom_tail_at_least(n, p, k_lo + 1)
        check(hoeffding_two_sided(t, n) >= float(exact_two),
              "Hoeffding must upper-bound the exact two-sided tail (n=%d)" % n)

    trials, n, t = 2000, 40, 0.25
    rng = Rng(31337)
    deviating = sum(1 for _ in range(trials)
                    if abs(sum(1 for _ in range(n) if rng.bernoulli(1, 2)) / n - 0.5) >= t)
    check(deviating / trials <= hoeffding_two_sided(t, n),
          "the empirical deviation rate must respect the Hoeffding bound")

    # ---- 7. amplification: majority vote over 2k+1 runs of a 2/3-correct subroutine
    for m in (5, 25, 101):
        exact = binom_tail_at_least(m, Fraction(1, 3), m // 2 + 1)
        check(Fraction(math.exp(-m / 18.0)).limit_denominator(10 ** 12) >= exact,
              "the majority amplification bound must hold at m=%d" % m)
        check(exact < Fraction(1, 3), "majority voting must beat a single run at m=%d" % m)
    rng = Rng(4242)
    trials = 3000
    correct = sum(1 for _ in range(trials) if sum(1 for _ in range(5) if rng.bernoulli(2, 3)) >= 3)
    observed = 1.0 - correct / trials
    exact5 = float(binom_tail_at_least(5, Fraction(1, 3), 3))
    check(abs(observed - exact5) <= 0.05,
          "observed majority failure rate %.4f must track the exact value %.4f" % (observed, exact5))
    check(observed <= float(binom_tail_at_least(5, Fraction(1, 3), 3)) + 0.05,
          "observed majority failure rate must respect the exact bound")
    check(observed <= math.exp(-5 / 18.0), "observed failure rate must respect the Hoeffding bound")
    check(correct / trials >= 0.6, "majority of five two-thirds-correct runs should succeed often")

    # ---- 8. labelling the bounds
    print("analytic mean n=64: %.3f comparisons" % float(expectation_closed_form(64)))
    print("4 n lg n bound n=64: %d comparisons" % (4 * 64 * math.ceil(math.log2(64))))
    print("chernoff(10, 0.5) = %.6f" % chernoff_upper(10.0, 0.5))
    print("PASS")


if __name__ == "__main__":
    main()
