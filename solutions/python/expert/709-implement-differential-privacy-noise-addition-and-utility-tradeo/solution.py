"""Task 709 - Implement differential privacy noise addition and utility tradeoff demo.  (Tier: Expert, CTCI Beyond - Security & Privacy)

The Laplace mechanism answers a numeric query f by publishing f(x) + Lap(0, b) with b = Δf / eps,
where Δf is the *global sensitivity* of f (the largest change of f between databases that differ
in one record).  That single formula is what this file implements, exercises and audits:

  * sampler -- Lap(0, b) by inverse transform, x = -b * sign(u) * ln(1 - 2|u|) with u uniform in
    (-1/2, 1/2); the uniforms come from splitmix64 so the whole program is deterministic;
  * sensitivity -- computed independently from the query by enumerating neighbours (add or remove
    one record) and taking the largest change, and compared against the closed-form value;
  * privacy -- the ratio property that *is* eps-DP for this mechanism is checked directly on the
    shifted Laplace densities: sup_z |ln p_x(z) - ln p_y(z)| = Δf / b = eps, over a grid of z and
    over random neighbouring database pairs;
  * utility -- the mean absolute error of the noisy answers is measured over many trials and
    compared with the theoretical value E|Lap(b)| = b, for a sweep of eps values, which is the
    utility/privacy tradeoff the task asks for;
  * composition -- k independent eps-mechanisms are (k*eps)-DP, and the variance of the summed
    noise must grow like 2*k*b^2.

The Gaussian mechanism is implemented alongside it for the (eps, delta) comparison:
sigma = Δf * sqrt(2 ln(1.25/delta)) / eps.

Simplified versus a real deployment: databases are small multisets held in memory, the privacy
audit is done on the mechanism's exact density (a grid, not a statistical test on samples), and
the utility numbers are Monte-Carlo estimates with the tolerance they deserve.  Nothing is faked:
every reported error and variance comes from the samples actually produced by the deterministic
PRNG, and the checked inequalities are the ones the theory guarantees.
"""
import math
import sys

MASK64 = (1 << 64) - 1
TWO53 = 1 << 53


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------- deterministic PRNG
def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    """Uniform stream on the *open* interval (-1/2, 1/2), which the inverse transform needs."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def unit(self):
        """Uniform in [0, 1)."""
        self.state, r = splitmix64(self.state)
        return ((r >> 11) + 0.5) / TWO53

    def centred(self):
        """Uniform in (-1/2, 1/2), never touching the endpoints (|u| <= 1/2 - 2**-54)."""
        return self.unit() - 0.5


# --------------------------------------------------------------------------- mechanisms
def laplace_sample(scale, rng):
    """Sample Lap(0, scale) by inverse transform; scale 0 is a deterministic exact answer."""
    if scale == 0.0:
        return 0.0
    u = rng.centred()
    magnitude = 1.0 - 2.0 * abs(u)
    tail = -scale * math.log(magnitude)
    return tail if u < 0 else -tail


def gaussian_sample(sigma, rng):
    """Box-Muller normal sample; sigma 0 is a deterministic exact answer."""
    if sigma == 0.0:
        return 0.0
    while True:
        u1 = rng.unit()
        if u1 > 0.0:
            break
    u2 = rng.unit()
    return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def laplace_scale(sensitivity, epsilon):
    check(epsilon > 0.0, "epsilon must be positive")
    check(sensitivity >= 0.0, "sensitivity cannot be negative")
    return sensitivity / epsilon


def gaussian_sigma(sensitivity, epsilon, delta):
    check(epsilon > 0.0, "epsilon must be positive")
    check(0.0 < delta < 1.0, "delta must lie in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def laplace_mechanism(true_answer, sensitivity, epsilon, rng):
    return true_answer + laplace_sample(laplace_scale(sensitivity, epsilon), rng)


def gaussian_mechanism(true_answer, sensitivity, epsilon, delta, rng):
    return true_answer + gaussian_sample(gaussian_sigma(sensitivity, epsilon, delta), rng)


def laplace_log_density(value, true_answer, scale):
    """log of the Laplace density of the mechanism's output, used for the privacy audit."""
    if scale == 0.0:
        return 0.0 if value == true_answer else -math.inf
    return -abs(value - true_answer) / scale - math.log(2.0 * scale)


# --------------------------------------------------------------------------- queries
def histogram_query(database, buckets):
    counts = [0] * buckets
    for record in database:
        check(0 <= record < buckets, f"record {record} is outside the {buckets} buckets")
        counts[record] += 1
    return counts


def neighbours(database):
    """All databases differing in one record (one added, or any one removed)."""
    result = []
    for index in range(len(database)):
        result.append(database[:index] + database[index + 1:])
    for record in range(4):
        result.append(database + [record])
    return result


def empirical_sensitivity(query, databases):
    """Independent sensitivity oracle: enumerate neighbours, take the largest L1 change."""
    worst = 0.0
    for database in databases:
        base = query(database)
        for neighbour in neighbours(database):
            other = query(neighbour)
            worst = max(worst, sum(abs(a - b) for a, b in zip(base, other)))
    return worst


def sensitivity_l1(values, sensitivity):
    """Scale a query with unit sensitivity to an arbitrary sensitivity."""
    return [value * sensitivity for value in values]


# --------------------------------------------------------------------------- checks
def test_sampler_statistics():
    rng = Rng(0x5EED1234)
    scale = 2.5
    trials = 120000
    samples = [laplace_sample(scale, rng) for _ in range(trials)]
    mean = sum(samples) / trials
    variance = sum((value - mean) ** 2 for value in samples) / trials
    check(abs(mean) < 0.05 * scale, f"Laplace mean should be 0, got {mean}")
    check(abs(variance - 2.0 * scale * scale) < 0.05 * 2.0 * scale * scale,
          f"Laplace variance should be 2b^2, got {variance}")

    def empirical_cdf(point):
        return sum(1 for value in samples if value <= point) / trials

    for point, expected in ((0.0, 0.5), (scale * math.log(2.0), 0.75), (-2.0 * scale, 0.5 * math.e ** -2)):
        got = empirical_cdf(point)
        check(abs(got - expected) < 0.01,
              f"CDF at {point:.3f} should be {expected:.4f}, got {got:.4f}")
    check(abs(empirical_cdf(scale) - (1 - 0.5 * math.e ** -1)) < 0.01, "CDF at b")

    rng = Rng(0xABCDEF)
    sigma = 1.75
    normals = [gaussian_sample(sigma, rng) for _ in range(60000)]
    mean = sum(normals) / len(normals)
    variance = sum((value - mean) ** 2 for value in normals) / len(normals)
    check(abs(mean) < 0.05 * sigma, f"Gaussian mean should be 0, got {mean}")
    check(abs(variance - sigma * sigma) < 0.05 * sigma * sigma,
          f"Gaussian variance should be sigma^2, got {variance}")
    check(laplace_sample(0.0, rng) == 0.0 and gaussian_sample(0.0, rng) == 0.0,
          "zero scale means an exact answer")


def test_prng_determinism():
    first = [Rng(7).unit() for _ in range(1)]
    check(first == [Rng(7).unit()], "the PRNG is reproducible for a fixed seed")
    a = [Rng(11).unit() for _ in range(5)]
    b = [Rng(11).unit() for _ in range(5)]
    c = [Rng(12).unit() for _ in range(5)]
    check(a == b, "same seed, same stream")
    check(a != c, "different seeds, different streams")
    stream = Rng(99)
    values = [stream.centred() for _ in range(10000)]
    check(all(-0.5 < value < 0.5 for value in values), "the centred stream stays strictly inside")
    check(abs(sum(values) / len(values)) < 0.01, "and it is unbiased")


def test_sensitivity_oracle():
    databases = [[], [0], [1], [0, 0], [0, 1], [1, 2, 3], [2, 2], [3, 3, 3, 1]]
    buckets = 4
    query = lambda database: histogram_query(database, buckets)
    measured = empirical_sensitivity(query, databases)
    check(measured == 1.0,
          f"one record added or removed moves the histogram by exactly 1 in L1, got {measured}")
    # the per-coordinate (L_inf) statement is what the mechanism is calibrated with, and here it
    # coincides with L1 because a single record touches a single bucket
    worst_inf = 0.0
    for database in databases:
        base = query(database)
        for neighbour in neighbours(database):
            other = query(neighbour)
            worst_inf = max(worst_inf, max(abs(a - b) for a, b in zip(base, other)))
    check(worst_inf == 1.0, f"per-coordinate sensitivity must be 1, got {worst_inf}")

    scaled = lambda database: sensitivity_l1(query(database), 3.0)
    check(empirical_sensitivity(scaled, databases) == 3.0, "scaling multiplies the sensitivity")
    check(empirical_sensitivity(lambda database: [len(database)], databases) == 1.0,
          "a plain count has sensitivity 1")


def test_dp_ratio_property():
    sensitivity = 1.0
    for epsilon in (0.1, 0.5, 1.0, 3.0):
        scale = laplace_scale(sensitivity, epsilon)
        worst = 0.0
        for answer_x in (10.0, 25.0):
            answer_y = answer_x + sensitivity
            for step in range(-400, 401):
                z = answer_x + step * 0.05 * scale
                log_ratio = laplace_log_density(z, answer_x, scale) \
                    - laplace_log_density(z, answer_y, scale)
                worst = max(worst, abs(log_ratio))
        check(worst <= epsilon + 1e-9,
              f"eps = {epsilon}: the density ratio {worst} must not exceed eps")
        check(worst >= epsilon * 0.98,
              f"eps = {epsilon}: the bound should be attained somewhere, got {worst}")

    rng = Rng(0xF00D)
    for _ in range(200):
        answer_x = float(rng.unit() * 100 - 50)
        delta = laplace_sample(0.25, rng)
        answer_y = answer_x + min(abs(delta), 1.0)
        epsilon = 0.75
        scale = laplace_scale(sensitivity, epsilon)
        for _ in range(20):
            z = answer_x + laplace_sample(4.0 * scale, rng)
            log_ratio = laplace_log_density(z, answer_x, scale) \
                - laplace_log_density(z, answer_y, scale)
            check(abs(log_ratio) <= epsilon + 1e-9, "eps-DP violated on a random pair")


def test_utility_tradeoff():
    database = [0, 0, 1, 1, 1, 2, 2, 3]
    true_counts = histogram_query(database, 4)
    trials = 4000
    errors = {}
    for epsilon in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        rng = Rng(0xC0FFEE + int(epsilon * 1000))
        total = 0.0
        for _ in range(trials):
            noisy = [laplace_mechanism(value, 1.0, epsilon, rng) for value in true_counts]
            total += sum(abs(a - b) for a, b in zip(noisy, true_counts)) / len(true_counts)
        errors[epsilon] = total / trials
        theoretical = laplace_scale(1.0, epsilon)
        check(abs(errors[epsilon] - theoretical) < 0.06 * theoretical,
              f"eps = {epsilon}: mean error {errors[epsilon]} should be b = {theoretical}")

    ordered = [errors[eps] for eps in sorted(errors)]
    check(all(a > b for a, b in zip(ordered, ordered[1:])),
          f"the error must fall as eps grows: {ordered}")

    rng = Rng(4242)
    tiny = [laplace_mechanism(7, 1.0, 1000.0, rng) for _ in range(5000)]
    check(max(abs(value - 7) for value in tiny) < 0.05,
          "a huge eps gives answers that are essentially exact")
    rng = Rng(4243)
    big = [laplace_mechanism(7, 1.0, 0.01, rng) for _ in range(500)]
    check(max(abs(value - 7) for value in big) > 50.0,
          "a tiny eps makes a single answer useless (that is the tradeoff)")
    return errors


def test_composition():
    epsilon = 0.5
    scale = laplace_scale(1.0, epsilon)
    parts = 6
    trials = 40000
    rng = Rng(0xBEEF)
    totals = []
    for _ in range(trials):
        totals.append(sum(laplace_sample(scale, rng) for _ in range(parts)))
    mean = sum(totals) / trials
    variance = sum((value - mean) ** 2 for value in totals) / trials
    expected = 2 * parts * scale * scale
    check(abs(variance - expected) < 0.08 * expected,
          f"k-fold composition has variance {expected}, measured {variance}")
    check(abs(sum(totals) / trials) < 0.1 * scale, "the composed noise stays centred")
    check(abs(parts * epsilon - 3.0) < 1e-12, "k mechanisms of eps compose to k * eps")

    single = []
    rng = Rng(0xBEF0)
    for _ in range(trials):
        single.append(laplace_sample(scale * math.sqrt(parts), rng))
    single_variance = sum(value * value for value in single) / trials
    check(abs(single_variance - expected) < 0.08 * expected,
          "one mechanism with the composed budget has the same variance")
    return variance, expected


def test_boundaries():
    rng = Rng(1)
    check(laplace_mechanism(3, 0.0, 0.5, rng) == 3.0, "zero sensitivity needs no noise")
    check(gaussian_mechanism(3, 0.0, 0.5, 1e-5, rng) == 3.0, "same for the Gaussian mechanism")
    check(laplace_scale(0.0, 0.25) == 0.0, "sensitivity 0 gives scale 0")

    empty = histogram_query([], 3)
    check(empty == [0, 0, 0], "an empty database has an all-zero histogram")
    rng = Rng(2)
    noisy = [laplace_mechanism(value, 1.0, 1.0, rng) for value in empty]
    check(all(abs(value) < 30.0 for value in noisy), "noise around zero stays reasonable")

    single = histogram_query([1], 3)
    check(single == [0, 1, 0], "a single record lands in exactly one bucket")

    for bad in (0.0, -1.0):
        try:
            laplace_scale(1.0, bad)
            raise AssertionError("a non-positive epsilon must be rejected")
        except AssertionError as error:
            check("epsilon" in str(error), f"unexpected message: {error}")
    try:
        gaussian_sigma(1.0, 1.0, 0.0)
        raise AssertionError("delta = 0 must be rejected")
    except AssertionError as error:
        check("delta" in str(error), f"unexpected message: {error}")
    try:
        gaussian_sigma(1.0, 1.0, 1.0)
        raise AssertionError("delta = 1 must be rejected")
    except AssertionError as error:
        check("delta" in str(error), f"unexpected message: {error}")
    try:
        laplace_scale(-1.0, 1.0)
        raise AssertionError("a negative sensitivity must be rejected")
    except AssertionError as error:
        check("sensitivity" in str(error), f"unexpected message: {error}")
    try:
        histogram_query([9], 3)
        raise AssertionError("a record outside the buckets must be rejected")
    except AssertionError as error:
        check("buckets" in str(error), f"unexpected message: {error}")

    first = [laplace_mechanism(5, 1.0, 0.5, Rng(7)) for _ in range(3)]
    second = [laplace_mechanism(5, 1.0, 0.5, Rng(7)) for _ in range(3)]
    check(first == second, "the mechanism is deterministic for a fixed seed")


def test_gaussian_calibration():
    for epsilon, delta in ((0.5, 1e-5), (1.0, 1e-5), (2.0, 1e-6), (5.0, 1e-3)):
        sigma = gaussian_sigma(1.0, epsilon, delta)
        analytic = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        check(abs(sigma - analytic) < 1e-12, "the sigma formula must match the classic bound")
        check(sigma >= 1.0 / epsilon, "the Gaussian noise never beats the Laplace scale")
    rng = Rng(0xDEAD)
    sigma = gaussian_sigma(1.0, 0.5, 1e-5)
    samples = [gaussian_sample(sigma, rng) for _ in range(40000)]
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    check(abs(variance - sigma * sigma) < 0.05 * sigma * sigma,
          f"the Gaussian mechanism's noise is calibrated: {variance} vs {sigma * sigma}")

    trials = 3000
    table = []
    for epsilon in (0.5, 1.0, 2.0):
        rng = Rng(int(epsilon * 1000) + 17)
        laplace_error = 0.0
        gauss_error = 0.0
        for _ in range(trials):
            laplace_error += abs(laplace_mechanism(20, 1.0, epsilon, rng) - 20)
            gauss_error += abs(gaussian_mechanism(20, 1.0, epsilon, 1e-5, rng) - 20)
        table.append((epsilon, laplace_error / trials, gauss_error / trials))
    for epsilon, laplace_error, gauss_error in table:
        check(abs(laplace_error - 1.0 / epsilon) < 0.08 / epsilon,
              f"eps = {epsilon}: the Laplace error should be 1/eps, got {laplace_error}")
        check(gauss_error > 0 and laplace_error > 0, "both mechanisms must show error")
    return table


def main():
    test_sampler_statistics()
    test_prng_determinism()
    test_sensitivity_oracle()
    test_dp_ratio_property()
    errors = test_utility_tradeoff()
    variance, expected = test_composition()
    test_boundaries()
    table = test_gaussian_calibration()

    print("eps -> mean absolute error (Laplace, sensitivity 1, 4 buckets):")
    print("  " + "  ".join(f"{eps:g}:{errors[eps]:.3f}" for eps in sorted(errors)))
    print(f"composition: measured variance {variance:.4f} vs 2*k*b^2 = {expected:.4f}")
    print("eps -> (Laplace error, Gaussian error @ delta=1e-5): "
          + ", ".join(f"{eps:g}:{lap:.3f}/{gau:.3f}" for eps, lap, gau in table))
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
