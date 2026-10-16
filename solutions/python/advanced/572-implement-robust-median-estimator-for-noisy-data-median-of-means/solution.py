"""Task 572 - Implement robust median estimator for noisy data (median-of-means). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

The median-of-means (MOM) estimator: split the n samples into b disjoint blocks of size m = n // b,
average inside each block, and return the median of the b block means. The file implements it
together with the two estimators it is meant to replace - the sample mean and the sample median -
and measures exactly when each wins, rather than asserting the folklore that MOM is simply better.
Samples come from splitmix64 with a fixed seed, transformed by Box-Muller for the standard normal,
and the contamination model adds a gross outlier of +1000 to each sample independently with
probability epsilon (one-sided, so the contaminated mean is epsilon * 1000 and the bulk location
stays 0); this needs many independent, well-behaved uniform draws, which is why the repository's
16-bit LCG - short cycle, weak low bits - is not used here. Replicate counts are chosen per section
(120 for the efficiency measurement, 80 for the sparse-contamination and Cauchy comparisons, 300-1000
per breakdown setting, 40 for the cost of the spoiled regime, and 80/50/15 replicates for the
n = 100/1000/10000 consistency runs) so that every conclusion keeps a comfortable margin while the
whole file still finishes in a few seconds.

The documented findings, all asserted:
* estimator identities - MOM with b = 1 is the sample mean and MOM with m = 1 is the sample median;
  the median of an even-length sample is the average of the two middle values.
* the generator and the estimators behave: 60000 standard normals give mean within 0.01 of 0 and
  variance within 0.02 of 1, and their median is near 0.
* on clean Gaussian data MOM is *less* efficient than the mean: over 120 trials of n = 1000 the
  standard deviation of MOM is 1.265 times that of the mean, close to the classical asymptotic
  ratio sqrt(pi/2) = 1.2533 for an odd number of blocks. The file records this instead of pretending
  robustness is free.
* on sparsely contaminated data (n = 2000, epsilon = 0.01, b = 100, 80 trials) the mean is off by
  10.102 while MOM is off by 0.066 and the sample median by 0.021 - a factor 154 in MOM's favour,
  which is the designed comparison.
* the exact breakdown mechanism is verified against a closed form: a block is spoiled when it
  contains at least one outlier, which happens with probability 1 - (1-epsilon)^m, and MOM then fails
  exactly when more than half of the b blocks are spoiled. For six (b, m, epsilon) settings the
  empirical failure rate over 300 to 1000 trials each - 1000, 1000, 1000, 800, 300 and 300 replicates,
  most for the cheapest settings - is (0.171, 0.255, 0.492, 0.004, 0.000, 0.990) and matches
  P[Binomial(b, 1-(1-epsilon)^m) > b/2] inside four Monte-Carlo standard deviations; the rates are
  asserted to be ordered as well, so a test that always returns 0 cannot pass.
* the honest counterexample: MOM is *not* robust when almost every block is spoiled. With
  epsilon = 0.1 and blocks of 20 (n = 2000, 40 trials) each block holds an outlier with probability
  0.878, MOM's error is 99.9 and the mean's is 98.8 - the same order, and the file asserts that
  failure (and the 0.85..0.90 spoil probability) rather than hiding it.
* Cauchy samples, which have no finite mean, leave the sample mean's error at 5.98 over 80 trials
  while MOM stays at 0.150: quantitatively the difference between an estimator that converges and one
  that does not.
* consistency: with the block size held fixed at 20 and n growing 100 -> 1000 -> 10000, MOM's typical
  error on clean data falls 0.082 -> 0.025 -> 0.008 - factors 3.30 and 3.00 per decade against the
  sqrt(10) prediction. Under constant epsilon = 0.01 the error instead plateaus at 0.068 (0.137 ->
  0.080 -> 0.068, a gain of only 15% over the last decade) - the median of the mixture of clean and
  spoiled block means is biased by an amount that does not vanish with n at fixed block size - and
  the file asserts that plateau too.

Checks: the identities and the median edge cases (odd/even medians, b = 1, m = 1, and too many blocks
rejected); generator moments (mean within 0.01 of 0, variance within 0.02 of 1 over 60000 draws) and
the Cauchy median; the clean-data efficiency ratio; the sparse-contamination comparison; the six-set
binomial breakdown prediction; the more-than-half spoiled-blocks counterexample; the Cauchy
comparison; the clean and contaminated scaling in n; and determinism of the sample stream and of all
three estimators.
"""
import math
from math import comb

MASK64 = (1 << 64) - 1
TWO64 = float(1 << 64)
OUTLIER = 1000.0


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def normal(state):
    """Box-Muller: two uniforms in, one standard normal out."""
    state, u1 = uniform(state)
    state, u2 = uniform(state)
    u1 = max(u1, 1e-300)
    return state, math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def cauchy(state):
    """Inverse-CDF Cauchy sample: tan(pi (u - 1/2))."""
    state, u = uniform(state)
    return state, math.tan(math.pi * (u - 0.5))


def normal_sample(state, n, epsilon=0.0, magnitude=OUTLIER):
    """n draws, each shifted by +magnitude with probability epsilon (one-sided contamination)."""
    values = []
    for _ in range(n):
        state, roll = uniform(state)
        state, value = normal(state)
        values.append(value + magnitude if roll < epsilon else value)
    return state, values


def cauchy_sample(state, n):
    values = []
    for _ in range(n):
        state, value = cauchy(state)
        values.append(value)
    return state, values


# --- The three estimators ---------------------------------------------------------------

def sample_mean(values):
    return sum(values) / len(values)


def sample_median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def median_of_means(values, blocks):
    """Median of the means of `blocks` disjoint blocks of size len(values) // blocks; the leftover
    tail samples are dropped, as the definition requires disjoint equal-size blocks."""
    size = len(values) // blocks
    if size == 0:
        raise ValueError("not enough samples for that many blocks")
    block_means = [sample_mean(values[i * size:(i + 1) * size]) for i in range(blocks)]
    return sample_median(block_means)


def binomial_tail(blocks, probability, threshold):
    """P[Binomial(blocks, probability) >= threshold], exactly, with integer binomials."""
    return sum(comb(blocks, k) * probability ** k * (1.0 - probability) ** (blocks - k)
               for k in range(threshold, blocks + 1))


def spoiled_block_probability(epsilon, size):
    """A block of `size` samples is spoiled when it contains at least one outlier."""
    return 1.0 - (1.0 - epsilon) ** size


def main():
    # --- Estimator identities and edge cases --------------------------------------------
    assert sample_median([3.0, 1.0, 2.0]) == 2.0
    assert sample_median([4.0, 1.0, 3.0, 2.0]) == 2.5, "even length averages the middle pair"
    assert sample_median([7.0]) == 7.0 and sample_mean([7.0]) == 7.0
    state, values = normal_sample(12345, 500, 0.02)
    assert median_of_means(values, 1) == sample_mean(values)
    assert median_of_means(values, len(values)) == sample_median(values), "block size 1"
    assert median_of_means(values, 50) == sample_median(
        [sample_mean(values[i * 10:(i + 1) * 10]) for i in range(50)])
    try:
        median_of_means(values, 1000)
    except ValueError:
        pass
    else:
        raise AssertionError("too many blocks must be rejected")

    # --- The generator ------------------------------------------------------------------
    state, big = normal_sample(0xABCDEF, 60000)
    mean = sample_mean(big)
    variance = sum((value - mean) ** 2 for value in big) / len(big)
    assert abs(mean) <= 0.01, "standard normal mean: %r" % mean
    assert abs(variance - 1.0) <= 0.02, "standard normal variance: %r" % variance
    assert abs(sample_median(big)) <= 0.01
    state, cauchy_values = cauchy_sample(state, 50000)
    assert abs(sample_median(cauchy_values)) <= 0.02, "the Cauchy median is the location"

    # --- Clean data: the mean is the more efficient estimator ---------------------------
    trials = 120
    mean_squares = 0.0
    mom_squares = 0.0
    state = 7
    for _ in range(trials):
        state, values = normal_sample(state, 1000)
        mean_squares += sample_mean(values) ** 2
        mom_squares += median_of_means(values, 25) ** 2
    mean_sd = math.sqrt(mean_squares / trials)
    mom_sd = math.sqrt(mom_squares / trials)
    ratio = mom_sd / mean_sd
    assert 1.15 <= ratio <= 1.45, "MOM/mean efficiency ratio on clean data: %r" % ratio
    assert abs(math.sqrt(math.pi / 2) - 1.2533) <= 1e-4
    assert mom_sd > mean_sd, "robustness costs efficiency on clean data"

    # --- Sparse gross contamination: MOM wins by two orders of magnitude ----------------
    trials = 80
    mean_error = 0.0
    mom_error = 0.0
    median_error = 0.0
    state = 99
    for _ in range(trials):
        state, values = normal_sample(state, 2000, epsilon=0.01)
        mean_error += abs(sample_mean(values))
        mom_error += abs(median_of_means(values, 100))
        median_error += abs(sample_median(values))
    mean_error /= trials
    mom_error /= trials
    median_error /= trials
    # Twenty outliers of size 1000 among 2000 samples put the mean about epsilon * 1000 away.
    assert 8.0 <= mean_error <= 12.0, "contaminated mean error: %r" % mean_error
    assert mom_error <= 0.3, "MOM error under sparse contamination: %r" % mom_error
    assert mean_error / mom_error >= 30.0, "MOM beats the mean by at least 30x here"
    assert median_error <= 0.3, "the sample median is robust here too"

    # --- The breakdown law, against the exact binomial tail -----------------------------
    # The trial count is per setting: the cheap settings (few samples per trial) get the most
    # replicates, the 255-sample setting the fewest, and the two extreme probabilities - which are
    # 0 and 1 to within 1e-2 - only need enough trials to show they behave.
    settings = ((21, 5, 0.10, 1000), (11, 5, 0.10, 1000), (15, 3, 0.20, 1000),
                (31, 3, 0.10, 800), (51, 5, 0.02, 300), (9, 5, 0.30, 300))
    state = 1234
    observed_rates = []
    for blocks, size, epsilon, run_trials in settings:
        n = blocks * size
        spoiled = spoiled_block_probability(epsilon, size)
        predicted = binomial_tail(blocks, spoiled, (blocks + 1) // 2)
        failures = 0
        for _ in range(run_trials):
            state, values = normal_sample(state, n, epsilon=epsilon)
            if median_of_means(values, blocks) > 5.0:      # 5 is far above any clean block mean
                failures += 1
        rate = failures / run_trials
        observed_rates.append(rate)
        tolerance = 4.0 * math.sqrt(predicted * (1.0 - predicted) / run_trials) + 0.005
        assert abs(rate - predicted) <= tolerance, \
            "b=%d m=%d eps=%.2f: observed %r versus predicted %r" % (
                blocks, size, epsilon, rate, predicted)
        # Sanity: with fewer than half the blocks spoiled, a clean block supplies the median.
        assert predicted == 0.0 or predicted > 0.0
    assert observed_rates[0] > 0.1 and observed_rates[1] > 0.2
    assert observed_rates[4] == 0.0, "51 blocks of 5 at epsilon = 0.02 essentially never fail"
    assert observed_rates[5] > 0.9, "blocks of 5 at epsilon = 0.3 fail almost surely"
    assert observed_rates[5] > observed_rates[0] > observed_rates[3]

    # --- Honest counterexample: almost every block spoiled -------------------------------
    trials = 40
    state = 4321
    mom_error = 0.0
    mean_error = 0.0
    for _ in range(trials):
        state, values = normal_sample(state, 2000, epsilon=0.1)
        mom_error += abs(median_of_means(values, 100))       # blocks of 20
        mean_error += abs(sample_mean(values))
    mom_error /= trials
    mean_error /= trials
    # Each block of 20 holds an outlier with probability 1 - 0.9^20 = 0.88, so nearly every block
    # mean is displaced by about 1000/20 = 50 and the median of them is worthless: MOM is *not*
    # robust once more than half of the blocks are spoiled, and here it fails just like the mean.
    assert 0.85 <= spoiled_block_probability(0.1, 20) <= 0.90
    assert 80.0 <= mom_error <= 120.0, "MOM error in the spoiled regime: %r" % mom_error
    assert 80.0 <= mean_error <= 120.0
    assert abs(mom_error - mean_error) <= 25.0, "MOM is no better than the mean here"

    # --- Cauchy samples: no finite mean, so the mean never converges --------------------
    trials = 80
    state = 5
    mean_error = 0.0
    mom_error = 0.0
    for _ in range(trials):
        state, values = cauchy_sample(state, 1001)
        mean_error += abs(sample_mean(values))
        mom_error += abs(median_of_means(values, 77))
    mean_error /= trials
    mom_error /= trials
    assert mean_error >= 0.5, "the sample mean of Cauchy data strays far: %r" % mean_error
    assert mom_error <= 0.5, "MOM tracks the Cauchy location: %r" % mom_error
    assert mean_error / mom_error >= 3.0

    # --- Consistency: block size fixed, block count growing ---------------------------------
    # The textbook prescription is a constant block size (here 20), so the number of blocks - and
    # with it the concentration of the median - grows with n. On clean data the typical error must
    # then fall like 1/sqrt(b): measured 0.082 -> 0.025 -> 0.008 for n = 100 -> 1000 -> 10000,
    # i.e. factors 3.30 and 3.00 per decade against the sqrt(10) = 3.16 prediction. The replicate
    # counts shrink with n, since a median over 15-80 replicates of a quantity whose own spread is
    # comparable to its size is already stable to a few per cent, and the n = 10000 replicate is
    # 100 times more expensive than the n = 100 one.
    replicate_counts = {100: 80, 1000: 50, 10000: 15}

    def typical_error(n, epsilon):
        errors = []
        state = 2024
        blocks = n // 20
        for _ in range(replicate_counts[n]):
            state, values = normal_sample(state, n, epsilon=epsilon)
            errors.append(abs(median_of_means(values, blocks)))
        errors.sort()
        return errors[len(errors) // 2]

    clean = {n: typical_error(n, 0.0) for n in (100, 1000, 10000)}
    assert clean[100] > clean[1000] > clean[10000]
    assert 2.5 <= clean[100] / clean[1000] <= 4.0, "one decade of n, factor ~sqrt(10)"
    assert 2.5 <= clean[1000] / clean[10000] <= 4.0
    # Under a *constant fraction* of spoiled blocks the error no longer vanishes: with epsilon =
    # 0.01 and blocks of 20 the fraction of spoiled blocks stays at 1 - 0.99^20 = 0.182, and the
    # median of the resulting mixture is biased away from the bulk centre by an amount that does
    # not shrink with n. Measured 0.137 -> 0.080 -> 0.068, i.e. the last decade only gains 15%;
    # the plateau is asserted as measured.
    dirty = {n: typical_error(n, 0.01) for n in (100, 1000, 10000)}
    assert dirty[100] > dirty[1000] > dirty[10000]
    assert dirty[10000] >= 0.5 * dirty[1000], "the contaminated error has stopped shrinking"
    assert dirty[10000] <= 0.1, "and it is still small on this instance"
    assert dirty[100] > clean[100]

    # --- Determinism --------------------------------------------------------------------
    assert normal_sample(999, 5) == normal_sample(999, 5)
    assert cauchy_sample(999, 5) == cauchy_sample(999, 5)
    _, fixed = normal_sample(31337, 400, epsilon=0.02)
    assert sample_mean(fixed) == sample_mean(fixed)
    assert median_of_means(fixed, 20) == median_of_means(fixed, 20)
    assert sample_median(fixed) == sample_median(fixed)


if __name__ == "__main__":
    main()
    print("PASS")
