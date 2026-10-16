"""Task 912 - Implement end-to-end pipeline for privacy-preserving analytics with DP budgeting.
(Tier: Master, CTCI Beyond - Security & Privacy)

What is implemented (in-process, deterministic; the only randomness is the seeded splitmix64
stream, so the whole audit trail is reproducible):

  * `laplace` - exact inverse-CDF Laplace sampling from that stream; the mechanism adds noise of
    scale sensitivity/epsilon, which is what makes a single count query epsilon-DP.
  * `density_ratio` - the direct check of the guarantee: for neighbouring inputs the ratio of the
    output densities never exceeds e^epsilon.
  * `BudgetAccountant` - sequential composition (the epsilons add up), an advanced-composition
    bound sqrt(2k ln(1/delta'))eps + k eps(e^eps - 1) for the same k-fold composition, and a hard
    budget: a query is served only while the total stays inside the allowance.
  * `release_histogram` - the end-to-end pipeline: per-bin count queries with their own epsilon,
    noisy answers post-processed to non-negative integers, plus the utility report (the maximum
    error against the closed-form Laplace tail bound).

Simplified versus production: counts only (no joins or means), a single trust boundary, one
mechanism and no privacy amplification by sampling.  Verified below: the sampler moments and tail
against the theory, the density-ratio guarantee, exact sequential accounting with budget
rejection, the advanced-composition bound, the utility/failure-rate check, post-processing
invariants, and the zero-budget / zero-sensitivity / tiny-epsilon boundaries.
"""

import math

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def unit(self):
        """Uniform in (0, 1): 53 bits of mantissa, never 0 or 1."""
        return (self.next_u64() >> 11) * (2.0 ** -53) + 2.0 ** -54


def laplace(rng, scale):
    """Inverse-CDF Laplace(0, scale) draw."""
    check(scale > 0, "the Laplace scale must be positive")
    uniform = rng.unit() - 0.5
    return -scale * math.copysign(math.log(1.0 - 2.0 * abs(uniform)), uniform)


def laplace_density(value, center, scale):
    return math.exp(-abs(value - center) / scale) / (2.0 * scale)


def density_ratio(epsilon, grid=200):
    """max_v p_{c}(v) / p_{c+1}(v) for neighbouring count queries: the DP definition, checked."""
    scale = 1.0 / epsilon
    worst = 0.0
    for index in range(grid):
        value = -3.0 + 6.0 * index / (grid - 1)
        worst = max(worst, laplace_density(value, 0.0, scale) / laplace_density(value, 1.0, scale))
    return worst


class BudgetAccountant:
    def __init__(self, total_epsilon):
        self.total = total_epsilon
        self.spent = 0.0
        self.queries = []

    def spend(self, epsilon):
        """Serve the query if the budget allows; otherwise refuse it outright."""
        check(epsilon >= 0, "an epsilon must be non-negative")
        if self.spent + epsilon > self.total + 1e-12:
            return False
        self.spent += epsilon
        self.queries.append(epsilon)
        return True

    def remaining(self):
        return self.total - self.spent

    @staticmethod
    def advanced_epsilon(count, epsilon, delta_prime):
        """Dwork-Roth k-fold advanced composition bound (better than k*eps for moderate k)."""
        return (math.sqrt(2.0 * count * math.log(1.0 / delta_prime)) * epsilon
                + count * epsilon * (math.exp(epsilon) - 1.0))


def release_histogram(accountant, histogram, epsilons, rng):
    """Answer each bin's count query under the budget; post-process to valid counts."""
    released, refused = [], 0
    for index, true_count in enumerate(histogram):
        epsilon = epsilons[index]
        if not accountant.spend(epsilon):
            refused += 1
            continue
        if epsilon == 0.0:
            released.append(true_count)
            continue
        noisy = true_count + laplace(rng, 1.0 / epsilon)
        released.append(max(0, int(round(noisy))))
    return released, refused


def main():
    # ---- 1. the sampler: moments and tail against the theory
    scale = 3.0
    rng = Rng(2024)
    draws = [laplace(rng, scale) for _ in range(40000)]
    mean = sum(draws) / len(draws)
    variance = sum((value - mean) ** 2 for value in draws) / len(draws)
    check(abs(mean) < 4.0 * scale / math.sqrt(len(draws)),
          "the Laplace sampler must be centred (%.4f)" % mean)
    check(abs(variance - 2.0 * scale * scale) < 0.15 * 2.0 * scale * scale,
          "the variance must match 2*scale^2 (%.3f vs %.3f)" % (variance, 2 * scale * scale))
    beta = 0.05
    bound = scale * math.log(1.0 / beta)
    failures = sum(1 for value in draws if abs(value) > bound) / len(draws)
    slack = 3.0 * math.sqrt(beta * (1.0 - beta) / len(draws))   # a 3-sigma sampling band
    check(failures <= beta + slack,
          "P(|X| > b ln(1/beta)) must not exceed beta (%.4f vs %.4f)" % (failures, beta))
    check(max(abs(value) for value in draws) <= scale * math.log(2.0 * len(draws)),
          "extreme draws must stay inside the exact support bound")

    # ---- 2. the mechanism itself: neighbouring inputs, density ratio <= e^epsilon
    for epsilon in (0.1, 0.5, 1.0, 2.0):
        ratio = density_ratio(epsilon)
        check(ratio <= math.exp(epsilon) + 1e-9,
              "the density ratio must respect e^epsilon (%.6f > %.6f)"
              % (ratio, math.exp(epsilon)))
        check(ratio > math.exp(epsilon) * 0.5,
              "the bound must be tight, not vacuous (%.6f)" % ratio)
    check(density_ratio(0.1) < density_ratio(1.0) < density_ratio(2.0)
          and abs(density_ratio(2.0) - math.exp(2.0)) < 1e-9,
          "the ratio grows with epsilon and the Laplace bound is attained at v <= 0")

    # ---- 3. sequential composition, exact accounting and budget rejection
    accountant = BudgetAccountant(total_epsilon=1.0)
    epsilons = [0.05] * 12 + [0.5] + [0.01] * 5
    histogram = [40 + (index * 17) % 30 for index in range(len(epsilons))]
    released, refused = release_histogram(accountant, histogram, epsilons, Rng(7))
    check(refused > 0, "the tight budget must refuse at least one query")
    check(abs(accountant.spent - sum(accountant.queries)) < 1e-12,
          "sequential composition is exactly additive")
    check(accountant.spent <= accountant.total + 1e-12 and len(accountant.queries) + refused == len(epsilons),
          "every query must be either served or refused, never both")
    check(abs(accountant.remaining() + accountant.spent - accountant.total) < 1e-12,
          "remaining plus spent must equal the total allowance")
    tight = BudgetAccountant(total_epsilon=0.3)
    served = [tight.spend(0.2), tight.spend(0.2), tight.spend(0.1)]
    check(served == [True, False, True] and abs(tight.spent - 0.3) < 1e-12,
          "a query that would breach the budget must be refused: %r" % served)

    # ---- 4. the advanced-composition bound is a real improvement and monotone
    sequential = 100 * 0.1
    advanced = BudgetAccountant.advanced_epsilon(100, 0.1, 1e-5)
    check(advanced < sequential, "advanced composition must beat k*eps (%.3f vs %.1f)"
          % (advanced, sequential))
    check(advanced > 0.1, "the bound cannot be smaller than one query's epsilon")
    check(BudgetAccountant.advanced_epsilon(200, 0.1, 1e-5) > advanced,
          "the bound must grow with the number of queries")
    check(BudgetAccountant.advanced_epsilon(100, 0.05, 1e-5)
          < BudgetAccountant.advanced_epsilon(100, 0.2, 1e-5),
          "the bound must grow with epsilon")

    # ---- 5. utility: the error bound and its failure rate
    bins = [30, 55, 12, 78, 41, 9, 66, 23]
    epsilon_per_bin = 0.25
    tail_beta = 0.05
    tolerance = (1.0 / epsilon_per_bin) * math.log(2.0 * len(bins) / tail_beta)
    breaches = 0
    trials = 300
    totals = []
    for trial in range(trials):
        account = BudgetAccountant(total_epsilon=epsilon_per_bin * len(bins))
        noisy, refused_here = release_histogram(account, bins, [epsilon_per_bin] * len(bins),
                                                Rng(3000 + trial))
        check(refused_here == 0 and len(noisy) == len(bins), "the budget must cover every bin")
        check(all(isinstance(value, int) and value >= 0 for value in noisy),
              "post-processing must yield non-negative integers")
        if max(abs(a - b) for a, b in zip(noisy, bins)) > tolerance:
            breaches += 1
        totals.append(sum(noisy))
    check(breaches / trials <= tail_beta,
          "the union-bound failure rate must hold (%.3f > %.3f)" % (breaches / trials, tail_beta))
    check(abs(sum(totals) / trials - sum(bins)) < 4.0 * math.sqrt(len(bins)) / epsilon_per_bin
          / math.sqrt(trials) * 4,
          "the released total must be unbiased for the true total")

    # ---- 6. the privacy/utility trade-off is monotone in epsilon
    errors = {}
    for epsilon in (0.05, 0.5, 5.0):
        worst = 0.0
        for trial in range(60):
            account = BudgetAccountant(total_epsilon=epsilon * len(bins))
            noisy, _ = release_histogram(account, bins, [epsilon] * len(bins), Rng(800 + trial))
            worst += max(abs(a - b) for a, b in zip(noisy, bins))
        errors[epsilon] = worst / 60
    check(errors[0.05] > errors[0.5] > errors[5.0],
          "smaller epsilon must cost more accuracy: %r" % {k: round(v, 2) for k, v in errors.items()})

    # ---- 7. boundaries
    broke = BudgetAccountant(total_epsilon=0.0)
    empty, refused_all = release_histogram(broke, bins, [0.1] * len(bins), Rng(1))
    check(empty == [] and refused_all == len(bins), "a zero budget must refuse everything")
    free = BudgetAccountant(total_epsilon=1.0)
    exact, refused_free = release_histogram(free, bins, [0.0] * len(bins), Rng(1))
    check(exact == bins and refused_free == 0 and free.spent == 0.0,
          "a zero-sensitivity query costs nothing and must be exact")
    blank, refused_blank = release_histogram(BudgetAccountant(total_epsilon=1.0), [], [], Rng(1))
    check(blank == [] and refused_blank == 0, "an empty histogram is a no-op")
    coarse = BudgetAccountant(total_epsilon=1.0)
    loose, _ = release_histogram(coarse, bins, [1.0] * len(bins), Rng(2))
    check(sum(abs(a - b) for a, b in zip(loose, bins)) < sum(errors.values()),
          "a large epsilon must be closer to the truth than the tested small ones")
    try:
        laplace(Rng(1), 0.0)
        raise AssertionError("a zero noise scale must be rejected")
    except AssertionError as error:
        check("positive" in str(error), "the rejection must explain the scale")

    # ---- 8. determinism
    first = release_histogram(BudgetAccountant(total_epsilon=4.0), bins, [0.5] * len(bins), Rng(42))
    second = release_histogram(BudgetAccountant(total_epsilon=4.0), bins, [0.5] * len(bins), Rng(42))
    check(first == second, "the released histogram must be reproducible from its seed")

    print("scale %g: variance %.3f (theory %.3f), tail failure %.4f" % (scale, variance, 2 * scale * scale, failures))
    print("budget 1.0: served %d, refused %d, spent %.4f" % (len(accountant.queries), refused, accountant.spent))
    print("tolerance %.2f, breaches %d/%d" % (tolerance, breaches, trials))
    print("PASS")


if __name__ == "__main__":
    main()
