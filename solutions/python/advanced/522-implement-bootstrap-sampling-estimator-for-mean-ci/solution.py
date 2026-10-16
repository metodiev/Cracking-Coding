"""Task 522 - Implement bootstrap sampling estimator for mean CI. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Non-parametric bootstrap confidence interval for the sample mean. Each of B resamples is a size-n
draw with replacement from the observed data - the indices come from a documented 64-bit LCG
(seed * 6364136223846793005 + 1442695040888963407 mod 2^64), advanced once per resampled index,
with the index taken from the top 31 bits of the state modulo n; the resample mean is accumulated
with an incremental mean so the whole procedure is O(B*n) time and O(1) extra memory per
resample. The interval is the percentile interval [Q(alpha/2), Q(1-alpha/2)] of the B bootstrap
means, with the quantile evaluated by linear interpolation on the (n-1)*p convention, alongside
the bootstrap bias (mean of the resample means minus the sample mean), the bootstrap standard
error, and the analytic normal interval mean +/- z * s / sqrt(n) with ddof = 1 and the
Abramowitz-Stegun rational z-quantile. Documented checks: the interpolated quantile on a
hand-computed 5-point vector, the bootstrap mean converging to the sample mean and the bootstrap
SE converging to s / sqrt(n) for a 400-point LCG sample, agreement of the percentile interval
with the analytic normal interval within a documented 12% of the analytic width, strict widening
with the confidence level, the width halving when n is quadrupled, the sample mean lying inside
every reported interval, exact degeneracy (zero width) on constant data and on n = 1, and
reproducibility for a repeated seed with a different seed giving a different resample stream.
"""

import math

MASK = (1 << 64) - 1


def lcg(seed):
    return (seed * 6364136223846793005 + 1442695040888963407) & MASK


def lcg_data(n, seed):
    """Triangular-ish sample in [0, 2): the sum of two uniform LCG draws."""
    data = []
    for _ in range(n):
        seed = lcg(seed)
        a = (seed >> 33) / float(1 << 31)
        seed = lcg(seed)
        b = (seed >> 33) / float(1 << 31)
        data.append(a + b)
    return data


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std_dev(values):
    """Sample standard deviation with the n-1 denominator; 0.0 for fewer than two points."""
    n = len(values)
    if n < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((v - average) ** 2 for v in values) / (n - 1))


def quantile(sorted_values, p):
    """Linear interpolation between order statistics, position (n-1)*p (the numpy default)."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    position = (len(sorted_values) - 1) * p
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def normal_z(p):
    """Two-sided standard-normal quantile by the Abramowitz-Stegun 26.2.23 rational fit."""
    if not 0.5 < p < 1.0:
        raise ValueError("p must lie in (0.5, 1)")
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def bootstrap_means(data, resamples, seed):
    """Bootstrap distribution of the mean, built index by index from the LCG stream."""
    n = len(data)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            seed = lcg(seed)
            total += data[(seed >> 33) % n]
        means.append(total / n)
    return means


def bootstrap_mean_ci(data, resamples, seed, level=0.95):
    """Percentile interval plus bias, bootstrap SE and the analytic normal interval."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    means = bootstrap_means(data, resamples, seed)
    ordered = sorted(means)
    alpha = 1.0 - level
    low = quantile(ordered, alpha / 2.0)
    high = quantile(ordered, 1.0 - alpha / 2.0)
    sample_mean = mean(data)
    bootstrap_mean = mean(means)
    standard_error = std_dev(means)
    z = normal_z(0.5 + level / 2.0)
    half_width = z * std_dev(data) / math.sqrt(len(data))
    return {"mean": sample_mean,
            "low": low,
            "high": high,
            "width": high - low,
            "bias": bootstrap_mean - sample_mean,
            "se": standard_error,
            "analytic_low": sample_mean - half_width,
            "analytic_high": sample_mean + half_width}


def main():
    # Hand-computed quantile: 5 sorted values, p = 0.25 sits exactly on the second one.
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(quantile(values, 0.0) - 1.0) <= 1e-15
    assert abs(quantile(values, 0.25) - 2.0) <= 1e-15
    assert abs(quantile(values, 0.5) - 3.0) <= 1e-15
    assert abs(quantile(values, 0.3) - 2.2) <= 1e-12
    assert abs(quantile(values, 1.0) - 5.0) <= 1e-15
    assert abs(quantile([7.0], 0.9) - 7.0) <= 1e-15
    # z-quantiles against textbook values (1.644854, 1.959964, 2.575829); the rational fit is
    # documented to be accurate to about 4.5e-4, so the checks use a 1e-3 tolerance.
    assert abs(normal_z(0.95) - 1.6448536) <= 1e-3
    assert abs(normal_z(0.975) - 1.959964) <= 1e-3
    assert abs(normal_z(0.995) - 2.575829) <= 1e-3
    assert abs(normal_z(0.975) - 1.959964) <= 5e-4

    # Bootstrap resamples are size-n draws from the data: every mean lies in the data range.
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    means = bootstrap_means(data, 200, 11)
    assert len(means) == 200
    assert all(min(data) - 1e-12 <= m <= max(data) + 1e-12 for m in means)
    assert bootstrap_means(data, 200, 11) == means, "same seed, same resample stream"
    assert bootstrap_means(data, 200, 12) != means, "different seed, different stream"
    # A constant sample has a degenerate bootstrap distribution.
    assert set(bootstrap_means([2.5] * 8, 50, 3)) == {2.5}

    # The bootstrap mean converges to the sample mean and the bootstrap SE to s / sqrt(n).
    data = lcg_data(400, 2024)
    n = len(data)
    sample_mean = mean(data)
    resamples = 4000
    means = bootstrap_means(data, resamples, 7)
    bootstrap_mean = mean(means)
    bootstrap_se = std_dev(means)
    analytic_se = std_dev(data) / math.sqrt(n)
    assert abs(bootstrap_mean - sample_mean) <= 0.01, (bootstrap_mean, sample_mean)
    assert abs(bootstrap_se - analytic_se) <= 0.15 * analytic_se, (bootstrap_se, analytic_se)

    # Percentile interval versus the analytic normal interval for a 400-point sample.
    report = bootstrap_mean_ci(data, resamples, 7, 0.95)
    analytic_width = report["analytic_high"] - report["analytic_low"]
    assert abs(report["width"] - analytic_width) <= 0.12 * analytic_width, report
    assert abs(report["se"] - analytic_se) <= 0.15 * analytic_se
    assert report["low"] < report["mean"] < report["high"], "interval covers the sample mean"
    assert report["analytic_low"] < report["mean"] < report["analytic_high"]
    assert abs(report["bias"]) <= 0.01, "mean of the resample means is essentially unbiased"
    assert abs(report["low"] - report["analytic_low"]) <= 0.12 * analytic_width
    assert abs(report["high"] - report["analytic_high"]) <= 0.12 * analytic_width

    # Wider confidence level -> strictly wider interval (same resample stream).
    previous = -1.0
    for level in (0.5, 0.8, 0.9, 0.95, 0.99):
        current = bootstrap_mean_ci(data, 1000, 7, level)
        assert current["width"] > previous, (level, current["width"])
        assert current["low"] < current["mean"] < current["high"], level
        previous = current["width"]
    assert previous > 0.0

    # Quadrupling n roughly halves the width (both intervals scale as 1 / sqrt(n)).
    for level in (0.9, 0.95):
        small = bootstrap_mean_ci(lcg_data(400, 2024), 1500, 7, level)["width"]
        large = bootstrap_mean_ci(lcg_data(1600, 2024), 1500, 7, level)["width"]
        ratio = small / large
        assert 1.4 <= ratio <= 2.6, (level, ratio)

    # Degeneracies: constant data and a single point give a zero-width interval and zero SE.
    constant = bootstrap_mean_ci([3.75] * 20, 100, 5, 0.95)
    assert constant["width"] == 0.0 and constant["se"] == 0.0 and constant["bias"] == 0.0
    assert constant["low"] == 3.75 and constant["high"] == 3.75
    single = bootstrap_mean_ci([42.0], 32, 5, 0.9)
    assert single["width"] == 0.0 and single["low"] == 42.0 and single["high"] == 42.0
    assert std_dev([42.0]) == 0.0

    # The interval shrinks toward the true mean as n grows, on data whose mean is known.
    shifted = [x - 1.0 for x in lcg_data(2000, 99)]        # triangular on [-1, 1), mean 0
    assert abs(mean(shifted)) <= 0.05
    widths = [bootstrap_mean_ci(shifted[:size], 800, 21, 0.95)["width"]
              for size in (100, 400, 1600)]
    assert widths[0] > widths[1] > widths[2], widths
    # Every interval contains the true mean 0.0 for these fixed seeds.
    for size in (100, 400, 1600):
        interval = bootstrap_mean_ci(shifted[:size], 800, 21, 0.95)
        assert interval["low"] < 0.0 < interval["high"], (size, interval)

    # Errors: empty sample and an out-of-range level / probability.
    for bad_call in (lambda: bootstrap_means([], 10, 1),
                     lambda: bootstrap_mean_ci([1.0, 2.0], 10, 1, 0.0),
                     lambda: bootstrap_mean_ci([1.0, 2.0], 10, 1, 1.0)):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    for bad_p in (-0.1, 1.5):
        try:
            quantile(values, bad_p)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for p = %r" % (bad_p,))

    # Monotone influence: shifting every observation by c shifts both intervals by c.
    base = bootstrap_mean_ci(data, 500, 7, 0.9)
    shifted_report = bootstrap_mean_ci([x + 10.0 for x in data], 500, 7, 0.9)
    assert abs(shifted_report["mean"] - (base["mean"] + 10.0)) <= 1e-12
    assert abs(shifted_report["low"] - (base["low"] + 10.0)) <= 1e-12
    assert abs(shifted_report["high"] - (base["high"] + 10.0)) <= 1e-12
    assert abs(shifted_report["se"] - base["se"]) <= 1e-12


if __name__ == "__main__":
    main()
    print("PASS")
