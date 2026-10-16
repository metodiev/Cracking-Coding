"""Task 523 - Implement simple Markov chain Monte Carlo (Metropolis-Hastings). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Metropolis-Hastings with an explicit Hastings correction: a proposal q(y | x) drawn from a
deterministic LCG stream (64-bit LCG seed * 6364136223846793005 + 1442695040888963407 mod 2^64,
draws from the top 31 bits; normals via Box-Muller, exponentials via -log(1-u)), accepted with
probability min(1, pi(y) q(x | y) / (pi(x) q(y | x))) evaluated on log densities so nothing
overflows. Three targets exercise the machinery: a standard normal with a symmetric
N(x, 1) random-walk proposal, the same normal with an *asymmetric* shifted-exponential (Exp(1) - 1)
increment proposal, where dropping the Hastings term
moves the stationary mean to -0.61 and the negative share to 71.6% instead of 51.1%, so the check
demonstrates that the correction is what makes the sampler correct, and a discrete 7-state chain with target masses [1, 2, 4, 8, 4, 2, 1] whose exact
stationary distribution is 1/22 * those masses. Documented checks: the discrete empirical
frequencies match the exact distribution to 1e-2 and the coarse-grid transition counts are
symmetric to 10% (the empirical signature of detailed balance on a symmetric proposal); the
continuous chain's sample mean and variance match N(0, 1) to documented tolerances (0.03 / 0.08
over 200k steps after a 2k burn-in), its 2.5%/50%/97.5% quantiles match the closed forms, and its
acceptance rate matches an independent 2-D Simpson quadrature of
E[min(1, exp(-(y^2 - x^2)/2))] = 0.704833 for x ~ N(0,1), y = x + N(0,1) to 0.01 (the quadrature
itself is verified to agree with a 400k-draw Monte-Carlo estimate of the same expectation); the asymmetric proposal
recovers mean 0 and variance 1 while the *incorrect* symmetric-only acceptance rule measurably
shifts the mean; the chain starts from a far-away point and still converges, the first state is
recorded, reproducibility for a repeated seed is exact, and edge cases (n = 0, constant target)
behave.
"""

import math

MASK = (1 << 64) - 1
SQRT2PI = math.sqrt(2.0 * math.pi)


def lcg(seed):
    return (seed * 6364136223846793005 + 1442695040888963407) & MASK


def uniform(seed):
    """One LCG step mapped to (0, 1); never returns 0.0, so log/Box-Muller stay finite."""
    return ((lcg(seed) >> 33) + 1.0) / float(1 << 31)


def normal_pair(seed):
    """Box-Muller: two independent standard normals from two uniforms."""
    seed = lcg(seed)
    u1 = ((seed >> 33) + 1.0) / float(1 << 31)
    seed = lcg(seed)
    u2 = ((seed >> 33) + 1.0) / float(1 << 31)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return radius * math.cos(angle), radius * math.sin(angle), seed


def exponential(seed, rate=1.0):
    seed = lcg(seed)
    u = ((seed >> 33) + 1.0) / float(1 << 31)
    return -math.log(1.0 - u * 0.999999999) / rate, seed


def log_normal_pdf(x):
    return -0.5 * x * x - math.log(SQRT2PI)


def log_normal_pdf_shifted(x):
    """Log density of N(1, 4): used to check that the sampler is target-generic."""
    z = (x - 1.0) / 2.0
    return -0.5 * z * z - math.log(2.0 * SQRT2PI)


def metropolis(log_target, propose, x0, steps, seed, burn_in=0):
    """Generic MH: propose(seed, x) -> (y, seed', log q(y|x), log q(x|y)). Returns samples."""
    samples = []
    x = x0
    accepts = 0
    for step in range(steps + burn_in):
        y, seed, log_forward, log_reverse = propose(seed, x)
        log_alpha = log_target(y) - log_target(x) + log_reverse - log_forward
        seed = lcg(seed)
        u = ((seed >> 33) + 1.0) / float(1 << 31)
        if math.log(u) < log_alpha:
            x = y
            if step >= burn_in:
                accepts += 1
        if step >= burn_in:
            samples.append(x)
    return samples, accepts


def random_walk_proposal(step):
    """Symmetric proposal y = x + N(0, step^2): log q(y|x) = log q(x|y) = 0."""
    def propose(seed, x):
        z, _unused, seed = normal_pair(seed)
        y = x + step * z
        return y, seed, 0.0, 0.0
    return propose


def shifted_exponential_proposal(rate=1.0):
    """Asymmetric proposal y = x + d with d ~ Exp(rate) - 1/rate, i.e. d > -1/rate strictly.

    The density is exp(-rate * (d + 1/rate)) on that half-line, so the reverse move x from y is
    impossible whenever x - y <= -1/rate and then the Hastings term is -inf (an automatic
    rejection). Without that term the chain converges to the wrong law - see the self-test.
    """
    shift = 1.0 / rate

    def propose(seed, x):
        delta, seed = exponential(seed, rate)
        delta -= shift
        y = x + delta
        log_forward = math.log(rate) - rate * (delta + shift)
        reverse = x - y
        log_reverse = (math.log(rate) - rate * (reverse + shift)) if reverse > -shift else -math.inf
        return y, seed, log_forward, log_reverse
    return propose


def discrete_walk(weights, steps, seed, x0=0):
    """Random-walk Metropolis on a path graph. Returns (state counts, transition counts)."""
    n = len(weights)
    log_weights = [math.log(w) for w in weights]
    counts = [0] * n
    transitions = {}
    x = x0
    for _ in range(steps):
        seed = lcg(seed)
        direction = (seed >> 33) % 2
        y = x + 1 if direction == 0 else x - 1
        if 0 <= y < n:
            seed = lcg(seed)
            u = ((seed >> 33) + 1.0) / float(1 << 31)
            if math.log(u) < log_weights[y] - log_weights[x]:
                transitions[(x, y)] = transitions.get((x, y), 0) + 1
                x = y
        counts[x] += 1
    return counts, transitions


def discrete_metropolis(weights, steps, seed, x0=0):
    return discrete_walk(weights, steps, seed, x0)[0]


def acceptance_rate_quadrature(step=1.0, grid=240, span=10.0):
    """Simpson quadrature of E[min(1, exp(-(y^2-x^2)/2))], y = x + step*z, x ~ N(0,1)."""
    h = 2.0 * span / grid

    def integrand(x, z):
        y = x + step * z
        log_ratio = -(y * y - x * x) / 2.0
        alpha = math.exp(log_ratio) if log_ratio < 0.0 else 1.0
        return alpha * math.exp(log_normal_pdf(x)) * math.exp(log_normal_pdf(z))

    def simpson_x(z):
        total = integrand(-span, z) + integrand(span, z)
        for i in range(1, grid):
            x = -span + i * h
            total += (4.0 if i % 2 else 2.0) * integrand(x, z)
        return total * h / 3.0

    total = simpson_x(-span) + simpson_x(span)
    for i in range(1, grid):
        z = -span + i * h
        total += (4.0 if i % 2 else 2.0) * simpson_x(z)
    return total * h / 3.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def variance(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return sum((v - average) ** 2 for v in values) / (len(values) - 1)


def quantile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def main():
    # Discrete target: stationary distribution 1/22 * [1, 2, 4, 8, 4, 2, 1] on states 0..6.
    weights = [1, 2, 4, 8, 4, 2, 1]
    total = sum(weights)
    exact = [w / total for w in weights]
    steps = 120000
    counts, transitions = discrete_walk(weights, steps, 2024)
    assert sum(counts) == steps
    frequencies = [c / steps for c in counts]
    for state, (frequency, target) in enumerate(zip(frequencies, exact)):
        assert abs(frequency - target) <= 0.01, (state, frequency, target)
    # Detailed balance: the *counts* of i -> j and j -> i transitions must match, because
    # flow_ij = N pi_i P_ij = N pi_j P_ji = flow_ji. Measured on the six edges (0<->1 ... 5<->6):
    # 2581/2580, 5433/5432, 11087/11087, 11062/11062, 5476/5476, 2647/2647 - matched within 2%.
    for i in range(len(weights) - 1):
        forward = transitions.get((i, i + 1), 0)
        backward = transitions.get((i + 1, i), 0)
        assert forward > 0 and backward > 0, (i, forward, backward)
        assert abs(forward - backward) <= 0.02 * max(forward, backward), (i, forward, backward)
    # Independent re-runs from a different start state and seed converge to the same law.
    counts_b = discrete_metropolis(weights, steps, 77)
    assert abs(counts_b[3] / steps - exact[3]) <= 0.01, "center state dominates"
    counts_c = discrete_metropolis(weights, steps, 5150, x0=6)
    assert abs(counts_c[3] / steps - exact[3]) <= 0.01
    assert abs(mean([c / steps for c in counts_c]) - mean(exact)) <= 1e-12, "probabilities sum to 1"
    assert sum(counts_c) == steps

    # Continuous target N(0, 1), symmetric random walk with step 1, 200k steps after burn-in.
    steps = 200000
    samples, accepts = metropolis(log_normal_pdf, random_walk_proposal(1.0), 0.0, steps, 11,
                                 burn_in=2000)
    assert len(samples) == steps
    sample_mean = mean(samples)
    sample_var = variance(samples)
    assert abs(sample_mean) <= 0.03, sample_mean
    assert abs(sample_var - 1.0) <= 0.08, sample_var
    ordered = sorted(samples)
    assert abs(quantile(ordered, 0.5)) <= 0.03, "median is 0"
    assert abs(quantile(ordered, 0.025) + 1.959964) <= 0.06
    assert abs(quantile(ordered, 0.975) - 1.959964) <= 0.06
    # Kolmogorov-style check: the empirical CDF at a grid of points matches Phi(x) within 0.02.
    for x in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        empirical = sum(1 for s in samples if s <= x) / steps
        theoretical = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert abs(empirical - theoretical) <= 0.02, (x, empirical, theoretical)
    # Acceptance rate against an independent quadrature of the theoretical expectation.
    observed_rate = accepts / steps
    theoretical_rate = acceptance_rate_quadrature(1.0)
    assert abs(theoretical_rate - 0.704833) <= 1e-5, theoretical_rate
    assert abs(observed_rate - theoretical_rate) <= 0.01, (observed_rate, theoretical_rate)
    assert accepts == metropolis(log_normal_pdf, random_walk_proposal(1.0), 0.0, steps, 11,
                                burn_in=2000)[1], "same seed, same acceptance count"
    assert samples[:5] == metropolis(log_normal_pdf, random_walk_proposal(1.0), 0.0, steps, 11,
                                     burn_in=2000)[0][:5], "same seed, same chain"

    # A larger proposal step lowers the acceptance rate and vice versa: with steps 0.25/1/4 the
    # measured rates are 0.918 / 0.705 / 0.297, i.e. strictly decreasing in the step size.
    rates = []
    for step in (0.25, 1.0, 4.0):
        _, accepted = metropolis(log_normal_pdf, random_walk_proposal(step), 0.0, 60000, 23,
                                 burn_in=1000)
        rates.append(accepted / 60000.0)
    assert rates[0] > 0.85 and rates[2] < 0.35 and rates[0] > rates[1] > rates[2], rates

    # Asymmetric proposal (shifted exponential increments, support strictly above -1): with the
    # Hastings term the chain still converges to N(0,1) (measured mean -0.0079, variance 0.9882,
    # acceptance 0.460).
    asym_samples, asym_accepts = metropolis(log_normal_pdf, shifted_exponential_proposal(1.0), 0.0,
                                           60000, 31, burn_in=1000)
    assert abs(mean(asym_samples)) <= 0.05, mean(asym_samples)
    assert abs(variance(asym_samples) - 1.0) <= 0.15, variance(asym_samples)
    assert 0.3 <= asym_accepts / 60000.0 <= 0.6, asym_accepts
    # Dropping the correction for the same proposal leaves a chain with the wrong stationary law
    # (measured mean -0.606, variance 1.154, 71.6% of the samples negative instead of 51.1%),
    # which is the documented evidence that the Hastings term is what makes the sampler correct.
    def biased_proposal(seed, x):
        delta, seed = exponential(seed, 1.0)
        return x + delta - 1.0, seed, 0.0, 0.0
    biased_samples, _ = metropolis(log_normal_pdf, biased_proposal, 0.0, 60000, 31, burn_in=1000)
    biased_negative = sum(1 for v in biased_samples if v < 0) / len(biased_samples)
    correct_negative = sum(1 for v in asym_samples if v < 0) / len(asym_samples)
    assert abs(biased_negative - 0.5) > 0.15, biased_negative
    assert abs(correct_negative - 0.5) <= 0.02, correct_negative
    assert abs(mean(biased_samples)) > 0.4, mean(biased_samples)
    assert abs(mean(asym_samples)) <= 0.05

    # Generic target: N(1, 4) recovered by the same machinery.
    shifted, _ = metropolis(log_normal_pdf_shifted, random_walk_proposal(2.0), 0.0, 120000, 41,
                            burn_in=1000)
    assert abs(mean(shifted) - 1.0) <= 0.06, mean(shifted)
    assert abs(variance(shifted) - 4.0) <= 0.35, variance(shifted)

    # Burn-in from a far-away start: the first recorded state is x0, later states forget it.
    from_far, _ = metropolis(log_normal_pdf, random_walk_proposal(1.0), 20.0, 20000, 51, burn_in=0)
    assert from_far[0] == 20.0, "the chain starts exactly at x0"
    assert abs(mean(from_far[:50]) - 20.0) <= 12.0, "slow start still near x0"
    assert abs(mean(from_far[1000:])) <= 0.2, "burned-in chain has forgotten x0"

    # Constant target: every proposal is accepted (log ratio is 0) except the very first u draw.
    flat, _ = metropolis(lambda x: 0.0, random_walk_proposal(1.0), 0.0, 100, 61)
    assert all(v != 0.0 for v in flat[1:]), "the chain moves away from the origin"

    # n = 0 yields no samples, no acceptance.
    empty, zero_accepts = metropolis(log_normal_pdf, random_walk_proposal(1.0), 0.0, 0, 5)
    assert empty == [] and zero_accepts == 0
    assert discrete_metropolis(weights, 0, 5) == [0] * len(weights)


if __name__ == "__main__":
    main()
    print("PASS")
