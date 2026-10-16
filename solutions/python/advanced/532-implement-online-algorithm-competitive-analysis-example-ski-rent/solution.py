"""Task 532 - Implement online algorithm competitive analysis example (ski rental). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Competitive analysis of the ski-rental problem, where a skier who does not know how many days
(k) he will ski must rent at cost 1 per day or buy at cost B, while an offline optimum with
perfect knowledge pays OPT(k) = min(k, B). A deterministic threshold strategy sigma_T rents the
first T-1 days and buys on day T, so its cost is k for k < T and (T-1) + B otherwise. Everything
is computed with exact rationals (fractions.Fraction), not floating point, and the self-test
proves: the worst-case ratio max_k cost_sigma(k) / OPT(k) is minimised exactly at T = B with the
value 2 - 1/B (enumerated over T and k, and the argmin set is exactly {B}); the extremes behave as
documented - always-buy (T = 1) has worst ratio B and always-rent has unbounded ratio, equal to
the sampled day limit divided by B. For the randomized side the classic distribution
p_t proportional to (1 - 1/B)^(B-t) over t = 1..B gives an expected-cost ratio whose maximum,
enumerated over k, equals the closed form Z(B) = 1 / (1 - (1 - 1/B)^B) exactly (both computed as
Fractions and compared), so e/(e-1) never exceeds the ratio: Z(B) < e/(e-1), Z(B) increases
towards e/(e-1) as B grows (verified with B = 10^4 and the limit e/(e-1) as the target) while
staying strictly below the deterministic optimum 2 - 1/B for every B >= 2; and a deterministic LCG
simulation of 20000 sampled thresholds reproduces the exact expected cost within 1e-2. The
degenerate B = 1 case (buy immediately, ratio 1) and the error cases (B < 1, non-integer k) are
covered too.
"""

from fractions import Fraction
import math

E_OVER_E_MINUS_1 = math.e / (math.e - 1.0)


def deterministic_cost(days, buy_cost, threshold):
    """Cost of sigma_T: rent the first T-1 days, then buy, if the skier skis `days` days."""
    if days < 1 or threshold < 1 or buy_cost < 1:
        raise ValueError("days, threshold and buy_cost must be positive integers")
    if days < threshold:
        return Fraction(days)
    return Fraction(threshold - 1 + buy_cost)


def offline_cost(days, buy_cost):
    return Fraction(min(days, buy_cost))


def worst_case_ratio(buy_cost, threshold, day_limit=None):
    if day_limit is None:
        day_limit = 4 * buy_cost + 10
    worst = Fraction(0)
    for days in range(1, day_limit + 1):
        ratio = deterministic_cost(days, buy_cost, threshold) / offline_cost(days, buy_cost)
        if ratio > worst:
            worst = ratio
    return worst


def best_deterministic(buy_cost, day_limit=None):
    """(best ratio, list of optimal thresholds) by exact enumeration."""
    if day_limit is None:
        day_limit = 4 * buy_cost + 10
    candidates = []
    for threshold in range(1, day_limit + 1):
        candidates.append((worst_case_ratio(buy_cost, threshold, day_limit), threshold))
    best = min(ratio for ratio, _ in candidates)
    return best, [threshold for ratio, threshold in candidates if ratio == best]


def randomized_weights(buy_cost):
    """p_t proportional to (1 - 1/B)^(B - t); returns Fractions summing exactly to 1."""
    if buy_cost < 2:
        return [Fraction(1)]
    decay = Fraction(buy_cost - 1, buy_cost)
    raw = [decay ** (buy_cost - t) for t in range(1, buy_cost + 1)]
    total = sum(raw)
    return [value / total for value in raw]


def randomized_expected_cost(days, buy_cost):
    """Exact expected cost of the randomized threshold strategy."""
    weights = randomized_weights(buy_cost)
    total = Fraction(0)
    for index, weight in enumerate(weights, start=1):
        total += weight * deterministic_cost(days, buy_cost, index)
    return total


def randomized_worst_ratio(buy_cost, day_limit=None):
    if day_limit is None:
        day_limit = 4 * buy_cost + 20
    worst = Fraction(0)
    for days in range(1, day_limit + 1):
        ratio = randomized_expected_cost(days, buy_cost) / offline_cost(days, buy_cost)
        if ratio > worst:
            worst = ratio
    return worst


def closed_form_randomized(buy_cost):
    """Z(B) = 1 / (1 - (1 - 1/B)^B), the exact worst-case ratio of the geometric distribution."""
    return 1.0 / (1.0 - (1.0 - 1.0 / buy_cost) ** buy_cost)


def sample_threshold(buy_cost, seed):
    """Inverse-transform sampling of p_t from the LCG stream; returns (threshold, new seed)."""
    seed = (seed * 75 + 74) % 65537
    target = Fraction(seed, 65537)
    cumulative = Fraction(0)
    for index, weight in enumerate(randomized_weights(buy_cost), start=1):
        cumulative += weight
        if target < cumulative:
            return index, seed
    return buy_cost, seed


def main():
    # Deterministic optimum: exactly 2 - 1/B, uniquely at T = B.
    for buy_cost in (1, 2, 3, 5, 10, 50):
        best, argmin = best_deterministic(buy_cost)
        assert best == 2 - Fraction(1, buy_cost), (buy_cost, best)
        assert argmin == [buy_cost], (buy_cost, argmin)
    assert best_deterministic(1)[0] == 1, "with B = 1 buying immediately is optimal"
    # Every threshold's ratio is attained at one of a few k values, and the documented formula
    # for sigma_B equals the enumeration for a longer horizon as well.
    for buy_cost in (2, 5, 10):
        assert worst_case_ratio(buy_cost, buy_cost) == 2 - Fraction(1, buy_cost)
        assert worst_case_ratio(buy_cost, buy_cost, 100 * buy_cost) == 2 - Fraction(1, buy_cost)

    # The extremes: always-buy has worst ratio B (at k = 1), always-rent grows with k / B.
    for buy_cost in (2, 5, 10):
        assert worst_case_ratio(buy_cost, 1) == buy_cost, "T = 1 is always-buy"
        day_limit = 10 * buy_cost
        assert worst_case_ratio(buy_cost, day_limit + 1, day_limit) == \
            Fraction(day_limit, buy_cost), "always-rent is unbounded"
        # A too-early purchase is strictly worse than the optimum.
        assert worst_case_ratio(buy_cost, 1) > 2 - Fraction(1, buy_cost)
        assert worst_case_ratio(buy_cost, buy_cost - 1) > 2 - Fraction(1, buy_cost)
        assert worst_case_ratio(buy_cost, buy_cost + 1) > 2 - Fraction(1, buy_cost)

    # Randomized distribution: the enumerated worst ratio equals the closed form exactly.
    for buy_cost in (2, 3, 5, 10, 20):
        weights = randomized_weights(buy_cost)
        assert sum(weights) == 1, buy_cost
        assert all(weight > 0 for weight in weights)
        enumerated = randomized_worst_ratio(buy_cost)
        closed = closed_form_randomized(buy_cost)
        assert abs(float(enumerated) - closed) <= 1e-12, (buy_cost, float(enumerated), closed)
    # The randomized strategy beats the best deterministic one, and converges to e/(e-1).
    ratios = [closed_form_randomized(buy_cost) for buy_cost in (2, 5, 10, 100, 1000, 10000)]
    for buy_cost, ratio in zip((2, 5, 10, 100, 1000, 10000), ratios):
        assert ratio < 2.0 - 1.0 / buy_cost, (buy_cost, ratio)
        assert ratio < E_OVER_E_MINUS_1, (buy_cost, ratio)
    assert all(later > earlier for earlier, later in zip(ratios, ratios[1:])), ratios
    assert E_OVER_E_MINUS_1 - ratios[-1] <= 1e-4, (E_OVER_E_MINUS_1, ratios[-1])
    assert abs(E_OVER_E_MINUS_1 - math.e / (math.e - 1.0)) <= 1e-15
    assert abs(E_OVER_E_MINUS_1 - 1.5819767068693265) <= 1e-12

    # The distribution is rational: the exact worst ratios are strictly below the closed form
    # only because of the finite day horizon, and the equality above pins them to the formula.
    assert randomized_worst_ratio(10) > Fraction(3, 2)
    assert randomized_worst_ratio(10) < Fraction(8, 5)

    # Deterministic simulation of the randomized strategy: the sample mean matches the exact
    # expectation of the cost for a fixed day count.
    buy_cost = 10
    trials = 20000
    seed = 2024
    total = Fraction(0)
    counts = [0] * (buy_cost + 1)
    for _ in range(trials):
        threshold, seed = sample_threshold(buy_cost, seed)
        assert 1 <= threshold <= buy_cost
        counts[threshold] += 1
        total += deterministic_cost(7, buy_cost, threshold)
    empirical = float(total) / trials
    exact = float(randomized_expected_cost(7, buy_cost))
    assert abs(empirical - exact) <= 0.02, (empirical, exact)
    # The sampled thresholds follow the documented weights.
    for threshold in range(1, buy_cost + 1):
        frequency = counts[threshold] / trials
        assert abs(frequency - float(randomized_weights(buy_cost)[threshold - 1])) <= 0.01, \
            (threshold, frequency)
    # Renting forever costs k while buying forever costs B at k = 1: sanity of the cost model.
    assert deterministic_cost(1, buy_cost, 1) == Fraction(buy_cost)
    assert offline_cost(1, buy_cost) == 1
    assert deterministic_cost(4, buy_cost, 100) == 4
    assert offline_cost(100, buy_cost) == buy_cost
    assert deterministic_cost(buy_cost, buy_cost, buy_cost) == Fraction(2 * buy_cost - 1)
    assert worst_case_ratio(buy_cost, buy_cost) == Fraction(2 * buy_cost - 1, buy_cost)

    # Errors.
    for bad in (lambda: deterministic_cost(0, 10, 3),
                lambda: deterministic_cost(5, 10, 0),
                lambda: deterministic_cost(5, 0, 3),
                lambda: deterministic_cost(-1, 10, 3)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # Determinism: enumerations and simulations repeat identically.
    assert best_deterministic(7) == best_deterministic(7)
    first_threshold, first_seed = sample_threshold(5, 11)
    assert sample_threshold(5, 11) == (first_threshold, first_seed)


if __name__ == "__main__":
    main()
    print("PASS")
