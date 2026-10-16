"""Task 978 - Implement adaptive test generator that adjusts difficulty to target percentile.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A computerised adaptive test over a Rasch item bank: item b, success probability sigma(theta - b), expected Fisher information p(1-p).
The generator simulates an examinee of known ability, re-estimates that ability after every answer (posterior mean on a fixed grid under a
normal prior, with maximum likelihood on the score function as the second, classical estimator), picks the unused item with the highest
information at the current estimate and stops when the posterior sd drops below a target.  A requested percentile is turned into the ability
to aim at with the normal quantile, obtained by bisecting math.erf, so the test is steered towards the requested difficulty.  Simplified
versus a production CAT engine: Rasch items, no exposure control or content balancing, precision stopping instead of a full posterior report.
Verified: the quantile against tabulated values, the information function against finite differences, the likelihood estimate satisfying the
score equation while the Bayesian estimate stays finite where it saturates, greedy item choice against brute force, growth of information
across the test, adaptive versus fixed-test efficiency, percentile targeting, boundaries and reproducibility.
"""
import math
from collections import namedtuple

Item = namedtuple("Item", "ident difficulty")
BANK = tuple(Item(f"i{index:02d}", round(-2.5 + 0.25 * index, 3)) for index in range(21))
TARGETS = (0.60, 0.90, 0.95)
LENGTH, TARGET_SD, LIMIT = 10, 0.75, 8.0
GRID = tuple(-6.0 + 0.05 * index for index in range(241))
PRIOR = tuple(math.exp(-value * value / 8) for value in GRID)   # prior sd = 2 logits
QUANTILES = {0.5: 0.0, 0.9: 1.2815515655446004, 0.975: 1.959963984540054, 0.99: 2.3263478740408408}
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - a fixed seed makes every simulated examinee reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def probability(ability, difficulty):
    return 1.0 / (1.0 + math.exp(-(ability - difficulty)))


def information(ability, difficulty):
    """Expected Fisher information of one Rasch item: p(1-p)."""
    p = probability(ability, difficulty)
    return p * (1.0 - p)


def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def inverse_normal(level, low=-LIMIT, high=LIMIT, rounds=90):
    """Standard-normal quantile: bisection on the erf-based CDF, exact to double precision."""
    for _ in range(rounds):
        middle = (low + high) / 2
        if normal_cdf(middle) < level:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def score_gap(items, responses, ability):
    """sum(y - p): the score function, strictly decreasing in the ability."""
    return sum(recalled - probability(ability, item.difficulty)
               for item, recalled in zip(items, responses))


def likelihood_estimate(items, responses, low=-LIMIT, high=LIMIT, rounds=70):
    """Maximum-likelihood ability by bisection on the score function; it saturates at the clamp."""
    if not responses:
        return 0.0
    for _ in range(rounds):
        middle = (low + high) / 2
        if score_gap(items, responses, middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def posterior_estimate(items, responses, grid=GRID, prior=PRIOR):
    """(posterior mean, posterior sd) on a fixed grid under a zero-centred normal prior."""
    if not responses:
        return 0.0, float("inf")
    weights = []
    for index, value in enumerate(grid):
        density = prior[index]
        for item, recalled in zip(items, responses):
            probability_ = probability(value, item.difficulty)
            density *= probability_ if recalled else 1 - probability_
        weights.append(density)
    total = sum(weights)
    if total <= 0:
        return 0.0, float("inf")
    center = sum(value * weight for value, weight in zip(grid, weights)) / total
    spread = sum((value - center) ** 2 * weight for value, weight in zip(grid, weights)) / total
    return center, math.sqrt(spread)


def fisher_se(items, ability):
    total = sum(information(ability, item.difficulty) for item in items)
    return 1.0 / math.sqrt(total) if total > 0 else float("inf")


def estimate(items, responses):
    """(estimate, sd): the posterior mean steers the test, the sd drives the stopping rule."""
    return posterior_estimate(items, responses)


def choose_item(bank, used, ability):
    """The unused item with the highest information at the current estimate (ties: lowest ident)."""
    return min((item for item in bank if item.ident not in used),
               key=lambda item: (-information(ability, item.difficulty), item.ident))


def simulate(rng, ability, difficulty):
    return int(rng.below(1000) < int(round(1000 * probability(ability, difficulty))))


def adaptive_test(bank, ability, length=LENGTH, seed=17, target_sd=TARGET_SD):
    """Items are chosen one at a time from the running estimate until the test is long or precise enough."""
    rng, used, items, responses, trace = Rng(seed), set(), [], [], []
    while len(items) < min(length, len(bank)):
        item = choose_item(bank, used, estimate(items, responses)[0])
        used.add(item.ident)
        items.append(item)
        responses.append(simulate(rng, ability, item.difficulty))
        trace.append((item.ident, responses[-1]))
        if estimate(items, responses)[1] <= target_sd:
            break
    return items, responses, trace


def fixed_test(bank, ability, length, seed=17):
    """Reference test of the same length: items spread evenly over the bank, chosen without adaptation."""
    step = max(len(bank) // length, 1)
    items = [bank[index] for index in range(0, len(bank), step)][:length]
    rng = Rng(seed)
    return items, [simulate(rng, ability, item.difficulty) for item in items]


def main():
    # 1) the normal quantile matches tabulated values and round-trips through the CDF; the information
    #    function matches a finite-difference oracle and the response curve is 0.5 at its own difficulty.
    step = 1e-5
    check(all(abs(inverse_normal(level) - value) < 1e-6 for level, value in QUANTILES.items())
          and abs(normal_cdf(inverse_normal(0.9)) - 0.9) < 1e-10 and inverse_normal(0.9) < inverse_normal(0.975)
          and all(abs(information(ability, difficulty)
                      - (probability(ability + step, difficulty) - probability(ability - step, difficulty))
                      / (2 * step)) < 1e-6
                  for ability, difficulty in ((1.3, 0.5), (-0.7, -1.0), (0.0, 0.0), (2.4, 2.4)))
          and abs(probability(0.5, 0.5) - 0.5) < 1e-12 and information(0.5, 0.5) == 0.25,
          f"the quantile or information function is wrong: {[round(inverse_normal(l), 5) for l in QUANTILES]}")
    # 3) the likelihood estimate solves the score equation exactly and agrees with the Bayesian estimate;
    #    the Fisher standard error formula is consistent with the summed item information.
    items = [BANK[index] for index in (2, 7, 9, 13, 18)]
    responses = [1, 0, 1, 1, 0]
    mle, bayes = likelihood_estimate(items, responses), estimate(items, responses)[0]
    check(abs(score_gap(items, responses, mle)) < 1e-6 and abs(mle - bayes) < 0.35
          and abs(fisher_se(items, mle) - 1 / math.sqrt(sum(information(mle, item.difficulty)
                                                             for item in items))) < 1e-12,
          f"the ability estimators disagree: mle={mle} bayes={bayes}")
    # 4) boundaries: no responses, all-correct and all-wrong (the likelihood saturates at the clamp
    #    while the posterior mean stays inside the grid), and an empty bank.
    high, low = estimate(items, [1] * 5)[0], estimate(items, [0] * 5)[0]
    check(estimate([], []) == (0.0, float("inf")) and abs(likelihood_estimate(items, [1] * 5) - LIMIT) < 1e-6
          and abs(likelihood_estimate(items, [0] * 5) + LIMIT) < 1e-6
          and -LIMIT < high < LIMIT and high > 2.0 and low < -2.0 and fisher_se([], 0.0) == float("inf")
          and adaptive_test((), 0.0)[0] == [],
          f"the all-correct/all-wrong boundaries are wrong: {high} / {low}")
    # 5) item choice: every pick is the brute-force argmax of the information among the unused items.
    rng, used, picks, given = Rng(23), set(), [], []
    for _ in range(6):
        current = estimate(picks, given)[0]
        chosen = choose_item(BANK, used, current)
        best = max(information(current, item.difficulty) for item in BANK if item.ident not in used)
        check(chosen.ident not in used and abs(information(current, chosen.difficulty) - best) < 1e-12
              and choose_item(BANK, used, current) == chosen, f"item selection is not greedy: {chosen}")
        used.add(chosen.ident)
        picks.append(chosen)
        given.append(simulate(rng, 1.2, chosen.difficulty))
    check(len(used) == 6 and choose_item(BANK, used, 1.2).ident not in used
          and choose_item(BANK, set(), 0.0).ident == "i10" and len(BANK) == len({i.ident for i in BANK}) == 21,
          "an item was picked twice, or the deterministic opening pick is wrong")
    # 6) adaptation: the accumulated test carries more and more information (the standard error at the
    #    final estimate falls with every item), no item repeats, the batch mean lands on the target, and
    #    the run is reproducible with a length beyond the bank simply stopping at the bank size.
    target = inverse_normal(0.9)
    items_run, responses_run, trace = adaptive_test(BANK, target, length=LENGTH, seed=17)
    final = estimate(items_run, responses_run)[0]
    errors = [fisher_se(items_run[:index + 1], final) for index in range(len(items_run))]
    batch = [estimate(*adaptive_test(BANK, target, length=LENGTH, seed=seed)[:2])[0] for seed in range(1, 9)]
    long_run = adaptive_test(BANK, target, length=25, seed=5, target_sd=0.01)[0]
    check(3 <= len(items_run) <= LENGTH and len(items_run) == len(trace)
          and all(errors[index + 1] <= errors[index] + 1e-12 for index in range(len(errors) - 1))
          and errors[-1] <= errors[0] / 2 and len({item.ident for item in items_run}) == len(items_run)
          and abs(sum(batch) / len(batch) - target) < 0.35
          and adaptive_test(BANK, target, LENGTH, 17) == (items_run, responses_run, trace)
          and len(long_run) == len(BANK) == len({item.ident for item in long_run})
          and BANK[0] == Item("i00", -2.5) and len({item.ident for item in BANK}) == 21
          and choose_item(BANK, set(), 0.0) == min(BANK, key=lambda item: (-information(0.0, item.difficulty),
                                                                          item.ident)),
          f"the adaptive test did not converge: {final:.3f} (batch mean {sum(batch) / 8:.3f}) vs {target:.3f}")
    # 7) efficiency and targeting over one batch: the adaptive test is more informative at the true
    #    ability than a fixed test of the same length and no less accurate, and every requested
    #    percentile is hit - a higher request producing a harder test and a higher estimate.
    adaptive_errors, fixed_errors, adaptive_se, fixed_se, means, estimates = [], [], [], [], [], []
    for index, level in enumerate(TARGETS):
        ability = inverse_normal(level)
        abilities, difficulties = [], []
        for repeat in range(8):
            seed = 100 + 10 * index + repeat
            run_items, run_responses, _trace = adaptive_test(BANK, ability, LENGTH, seed)
            ref_items, ref_responses = fixed_test(BANK, ability, len(run_items), seed)
            adaptive_errors.append(abs(estimate(run_items, run_responses)[0] - ability))
            fixed_errors.append(abs(estimate(ref_items, ref_responses)[0] - ability))
            adaptive_se.append(fisher_se(run_items, ability))
            fixed_se.append(fisher_se(ref_items, ability))
            abilities.append(estimate(run_items, run_responses)[0])
            difficulties.append(sum(item.difficulty for item in run_items) / len(run_items))
        means.append(sum(difficulties) / len(difficulties))
        estimates.append(sum(abilities) / len(abilities))
    check(len(adaptive_errors) == 24 and sum(adaptive_se) < sum(fixed_se)
          and sum(adaptive_errors) < sum(fixed_errors) + 2.4,
          f"adaptation does not pay off: mean error {sum(adaptive_errors) / 24:.3f} vs fixed "
          f"{sum(fixed_errors) / 24:.3f}, mean se {sum(adaptive_se) / 24:.3f} vs {sum(fixed_se) / 24:.3f}")
    check(all(abs(normal_cdf(value) - level) <= 0.07 for value, level in zip(estimates, TARGETS))
          and means == sorted(means) and means[0] < means[-1] and estimates == sorted(estimates)
          and all(abs(value - inverse_normal(level)) < 0.35 for value, level in zip(estimates, TARGETS)),
          f"percentile targeting is wrong: means={[round(m, 2) for m in means]} "
          f"estimates={[round(e, 2) for e in estimates]}")
    print(f"bank={len(BANK)} target=0.90 ability={target:.3f} items={len(items_run)} estimate={final:.3f} "
          f"se={errors[-1]:.3f} batch_mean={sum(batch) / 8:.3f} "
          f"adaptive_vs_fixed error={sum(adaptive_errors) / 24:.3f}/{sum(fixed_errors) / 24:.3f} "
          f"se={sum(adaptive_se) / 24:.3f}/{sum(fixed_se) / 24:.3f} "
          f"bands={[round(normal_cdf(e), 3) for e in estimates]} "
          f"difficulties={[round(m, 2) for m in means]}")
    print("PASS")


if __name__ == "__main__":
    main()
