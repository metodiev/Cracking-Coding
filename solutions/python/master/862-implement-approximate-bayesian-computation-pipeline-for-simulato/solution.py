"""Task 862 - Implement approximate Bayesian computation pipeline for simulator-based inference.  (Tier: Master, Beyond — Machine Learning & AI)

An ABC pipeline for a simulator whose likelihood is only available by running it: the prior
is a finite grid of parameters, the simulator turns a parameter into a summary statistic,
and inference is rejection ABC with an optional local-linear regression adjustment.  Because
the summary is discrete and the prior is a finite grid, the *exact* posterior is available by
enumeration, so every approximation can be measured against the truth instead of assumed.

What is simplified: the simulator is a binomial draw (the canonical tractable stand-in for a
black-box generative model), the prior is a uniform grid over nine parameter values, and the
distance is the absolute difference of the summary statistics.  The "randomness" is a seeded
splitmix64 stream, so runs are reproducible; there are no threads, no clock and no
dependencies.

Invariants verified:
  * exact-match ABC: with tolerance zero the accepted parameters are the ones whose simulated
    summary equals the observation, so the ABC posterior converges to the exact posterior -
    checked against the enumerated posterior, and against the analytic likelihood ratio;
  * the acceptance rate is predicted analytically (the prior-averaged probability that the
    simulator lands within the tolerance of the observation) and matches the measured rate;
  * the approximation degrades gracefully: the KL divergence from the exact posterior grows
    monotonically as the tolerance loosens, and an enormous tolerance recovers the prior
    exactly (every parameter is accepted with probability one);
  * regression adjustment reduces the error in the posterior mean compared with plain
    rejection ABC, which is the reason it exists;
  * an impossible observation (zero probability under every parameter) is reported as such
    instead of silently returning an empty posterior; the estimator tightens as the simulation
    budget grows; a second family of observations agrees with the exact posterior; a one-sided
    acceptance window never accepts more than the symmetric one; and reproducibility holds
    throughout.
"""

import math

MASK64 = (1 << 64) - 1
TRIALS = 12
GRID = tuple(round(0.1 * index, 2) for index in range(1, 10))     # the prior support


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return self.next_u64() / float(1 << 64)

    def below(self, n):
        return self.next_u64() % n


def binomial_probability(successes, trials, probability):
    if successes < 0 or successes > trials:
        return 0.0
    return (math.comb(trials, successes) * probability ** successes
            * (1.0 - probability) ** (trials - successes))


def simulate(parameter, rng):
    """The black-box simulator: `trials` Bernoulli draws, summarised by the success count."""
    successes = 0
    for _ in range(TRIALS):
        if rng.uniform() < parameter:
            successes += 1
    return successes


def exact_posterior(observation):
    """Enumerate the posterior over the grid: prior * likelihood, normalised."""
    weights = [binomial_probability(observation, TRIALS, parameter) for parameter in GRID]
    total = sum(weights)
    if total <= 0.0:
        return None
    return [weight / total for weight in weights]


def exact_acceptance(tolerance, observation):
    """Analytic probability that a single (parameter, simulation) pair is accepted."""
    total = 0.0
    for parameter in GRID:
        probability = sum(binomial_probability(count, TRIALS, parameter)
                          for count in range(max(0, observation - tolerance),
                                             min(TRIALS, observation + tolerance) + 1))
        total += probability
    return total / len(GRID)


def rejection_abc(observation, tolerance, simulations, seed, low=None):
    """Rejection ABC: sample the prior, simulate, keep the (parameter, summary) pairs whose
    summary is close to the observation.  `low=-tolerance, high=tolerance` is the usual
    symmetric window; passing `low=0` gives the one-sided window that biased ABC studies use.
    """
    rng = SplitMix64(seed)
    bottom = -tolerance if low is None else low
    accepted = []
    for _ in range(simulations):
        parameter = GRID[rng.below(len(GRID))]
        summary = simulate(parameter, rng)
        if bottom <= summary - observation <= tolerance:
            accepted.append((parameter, summary))
    return accepted, simulations


def regression_adjusted(accepted, observation):
    """Local-linear regression adjustment (Beaumont et al.): regress the parameter on the
    summary difference of the accepted sample and shift each draw by the fitted line."""
    if len(accepted) < 3:
        return [parameter for parameter, _summary in accepted]
    differences = [summary - observation for _parameter, summary in accepted]
    parameters = [parameter for parameter, _summary in accepted]
    mean_difference = sum(differences) / len(differences)
    mean_parameter = sum(parameters) / len(parameters)
    denominator = sum((value - mean_difference) ** 2 for value in differences)
    if denominator == 0.0:
        return list(parameters)
    slope = sum((value - mean_difference) * (parameter - mean_parameter)
                for value, parameter in zip(differences, parameters)) / denominator
    # the line is evaluated at the *observed* summary (difference zero), which is what
    # removes the bias introduced by the acceptance window
    return [parameter - slope * value for value, parameter in zip(differences, parameters)]


def posterior_from_samples(samples):
    """Snap a sample of (possibly adjusted) parameters back onto the prior grid."""
    weights = {parameter: 0.0 for parameter in GRID}
    for value in samples:
        closest = min(GRID, key=lambda parameter: (abs(parameter - value), parameter))
        weights[closest] += 1.0
    total = sum(weights.values())
    if total <= 0.0:
        return None
    return [weights[parameter] / total for parameter in GRID]


def kl_divergence(left, right):
    total = 0.0
    for a, b in zip(left, right):
        if a > 0.0 and b > 0.0:
            total += a * math.log(a / b)
        elif a > 0.0:
            return float("inf")
    return total


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. the model and its exact posterior ----------------------------
    observation = 8
    exact = exact_posterior(observation)
    check(exact is not None and abs(sum(exact) - 1.0) < 1e-12,
          "the enumerated posterior must be a distribution")
    likelihoods = [binomial_probability(observation, TRIALS, parameter) for parameter in GRID]
    best = GRID[likelihoods.index(max(likelihoods))]
    check(best == 0.7, "the maximum-likelihood grid point must be 0.7, got %s" % best)
    peak = exact.index(max(exact))
    check(peak == 6, "the posterior must peak at 0.7 (grid index %d)" % peak)
    check(all(exact[index] <= exact[index + 1] for index in range(peak)),
          "the posterior must rise towards the peak")
    check(all(exact[index] >= exact[index + 1] for index in range(peak, len(exact) - 1)),
          "the posterior must fall away after the peak")
    check(exact[0] < 1e-3, "parameters far from the observation must be ruled out")
    check(exact[8] < exact[7] < exact[6], "the far end of the grid must be suppressed")
    print("exact posterior: %s (ML grid point %.2f)"
          % ([round(value, 4) for value in exact], best))

    # ---- 2. exact-match ABC converges to the exact posterior -------------
    matched, attempts = rejection_abc(observation, 0, 400000, 20240501)
    check(attempts == 400000, "the pipeline must run every requested simulation")
    check(len(matched) > 500, "exact matches must be found (%d)" % len(matched))
    check(all(abs(summary - observation) <= 0 for _parameter, summary in matched),
          "every exact-match draw must reproduce the observation exactly")
    empirical = posterior_from_samples([parameter for parameter, _s in matched])
    deviation = max(abs(a - b) for a, b in zip(empirical, exact))
    check(deviation < 0.03,
          "tolerance-zero ABC must approach the exact posterior (max deviation %.4f)" % deviation)
    print("tolerance 0: %d accepted of %d simulations, max |ABC - exact| = %.4f"
          % (len(matched), attempts, deviation))

    # ---- 3. the acceptance rate is predictable ---------------------------
    for tolerance in (0, 1, 2):
        accepted, total = rejection_abc(observation, tolerance, 60000, 777)
        predicted = exact_acceptance(tolerance, observation)
        observed = len(accepted) / total
        check(abs(observed - predicted) < 0.01,
              "tolerance %d: acceptance %.4f != predicted %.4f"
              % (tolerance, observed, predicted))
    print("acceptance rates: %s"
          % {tolerance: (round(exact_acceptance(tolerance, observation), 4))
             for tolerance in (0, 1, 2, 3)})

    # ---- 4. loosening the tolerance degrades gracefully ------------------
    divergences = []
    for tolerance in (0, 1, 2, 3, 6):
        accepted, _total = rejection_abc(observation, tolerance, 120000, 31337)
        approximate = posterior_from_samples([parameter for parameter, _s in accepted])
        divergences.append((tolerance, kl_divergence(approximate, exact)))
    check(all(divergences[index][1] <= divergences[index + 1][1] + 1e-9
              for index in range(len(divergences) - 1)),
          "the KL divergence must grow with the tolerance: %s"
          % [(t, round(value, 4)) for t, value in divergences])
    check(divergences[-1][1] > divergences[0][1],
          "a huge tolerance must be strictly worse than an exact match")
    prior = [1.0 / len(GRID)] * len(GRID)
    huge, _total = rejection_abc(observation, 99, 40000, 5)
    recovered = posterior_from_samples([parameter for parameter, _s in huge])
    check(max(abs(left - right) for left, right in zip(recovered, prior)) < 0.02,
          "an enormous tolerance must recover the prior (max deviation %.4f)"
          % max(abs(left - right) for left, right in zip(recovered, prior)))
    print("tolerance sweep: %s, and the prior is recovered at tolerance 99"
          % [(t, round(value, 4)) for t, value in divergences])

    # ---- 5. regression adjustment ----------------------------------------
    def posterior_mean(samples):
        return sum(samples) / len(samples) if samples else 0.0

    exact_mean = sum(parameter * weight for parameter, weight in zip(GRID, exact))
    plain_error = adjusted_error = 0.0
    for seed in range(6):
        # a one-sided acceptance window biases the plain estimate; the adjustment removes it
        accepted, _total = rejection_abc(observation, 2, 60000, 900 + seed, low=0)
        plain = posterior_mean([parameter for parameter, _s in accepted])
        adjusted = posterior_mean(regression_adjusted(accepted, observation))
        plain_error += abs(plain - exact_mean)
        adjusted_error += abs(adjusted - exact_mean)
    check(plain_error > 0.0, "the one-sided window must bias the plain estimate")
    check(adjusted_error < plain_error,
          "the regression adjustment must reduce the bias (%.4f vs %.4f)"
          % (adjusted_error, plain_error))
    symmetric, _total = rejection_abc(observation, 2, 60000, 4242)
    check(len(regression_adjusted(symmetric, observation)) == len(symmetric),
          "the adjustment must keep every accepted draw")
    print("adjustment: one-sided bias plain %.4f vs adjusted %.4f (exact mean %.4f)"
          % (plain_error / 6, adjusted_error / 6, exact_mean))

    # ---- 6. edge cases ---------------------------------------------------
    impossible = exact_posterior(TRIALS + 5)
    check(impossible is None, "an impossible observation must have no posterior")
    empty, attempts = rejection_abc(TRIALS + 5, 0, 5000, 11)
    check(empty == [] and attempts == 5000,
          "an impossible observation must accept nothing but still run the simulations")
    check(posterior_from_samples([]) is None,
          "an empty sample must not be turned into a posterior")
    everything, _total = rejection_abc(observation, 99, 1000, 5)
    check(len(everything) == 1000, "an enormous tolerance must accept every simulation")
    check(regression_adjusted([(0.5, 8)], observation) == [0.5],
          "too few samples must be returned unchanged")
    check(rejection_abc(observation, 0, 0, 1) == ([], 0),
          "zero simulations must be a no-op")
    print("edge cases: impossible observation, empty sample, zero simulations")

    # ---- 7. reproducibility ---------------------------------------------
    again, _total = rejection_abc(observation, 1, 5000, 31337)
    once, _total = rejection_abc(observation, 1, 5000, 31337)
    check(again == once, "the same seed must give the same accepted sample")
    other, _total = rejection_abc(observation, 1, 5000, 31338)
    check(other != again, "a different seed must give a different sample")
    print("determinism: identical accepted samples for the same seed")

    # ---- 8. the estimator improves with more simulations -----------------
    deviations = []
    for simulations in (5000, 50000, 400000):
        accepted, _total = rejection_abc(observation, 0, simulations, 20240501)
        estimate = posterior_from_samples([parameter for parameter, _s in accepted])
        deviations.append((simulations, len(accepted),
                           max(abs(a - b) for a, b in zip(estimate, exact))))
    check(deviations[0][2] > deviations[-1][2],
          "more simulations must mean a closer estimate: %s"
          % [(step, round(error, 4)) for step, _count, error in deviations])
    check(deviations[-1][2] < 0.02,
          "the largest run must be inside two points of the exact posterior: %.4f"
          % deviations[-1][2])
    check(all(deviations[index][1] < deviations[index + 1][1]
              for index in range(len(deviations) - 1)),
          "the accepted count must grow with the simulation budget")

    # ---- 9. a second observation and the one-sided window ----------------
    for other_observation in (4, 6, 11):
        truth = exact_posterior(other_observation)
        accepted, _total = rejection_abc(other_observation, 0, 400000, 101010)
        estimate = posterior_from_samples([parameter for parameter, _s in accepted])
        check(max(abs(a - b) for a, b in zip(estimate, truth)) < 0.03,
              "observation %d: tolerance-zero ABC must match the exact posterior"
              % other_observation)
        nearest = min(GRID, key=lambda parameter: (abs(parameter - other_observation / TRIALS),
                                                   parameter))
        check(GRID[truth.index(max(truth))] == nearest,
              "observation %d: the exact posterior must peak at the grid point nearest the "
              "observed success rate (%.2f vs %.2f)"
              % (other_observation, GRID[truth.index(max(truth))], nearest))
    one_sided = 0
    for tolerance in (0, 1, 2):
        symmetric_accepted, total = rejection_abc(observation, tolerance, 40000, 606)
        biased_accepted, _same = rejection_abc(observation, tolerance, 40000, 606, low=0)
        if len(biased_accepted) <= len(symmetric_accepted):
            one_sided += 1
        check(len(biased_accepted) / total <= exact_acceptance(tolerance, observation) + 0.01,
              "the one-sided window must not accept more than the analytic rate")
    check(one_sided == 3,
          "a one-sided window can never accept more than the symmetric one")
    print("observations 4, 6, 8, 11 agree with the exact posterior; the one-sided window is "
          "never wider (%s)" % {step: round(deviation, 4) for step, _c, deviation in deviations})
    print("PASS")


if __name__ == "__main__":
    main()
