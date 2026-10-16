"""Task 906 - Implement multi-fidelity simulation orchestration with surrogate models.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; the sample design is a van-der-Corput sequence
rather than a PRNG, so every number below is exactly reproducible):

  * `f_high` / `f_low` - a "physical" response evaluated at two fidelities: an expensive truth
    f_h and a cheap surrogate f_l = f_h - bias(x) with a smooth bias, which is the usual
    setting in which multi-fidelity methods help.
  * `quadrature` - the reference answer: midpoint quadrature of the truth on a fine grid; its
    own convergence (n vs 2n) is checked, so the oracle is not trusted blindly.
  * `mf_estimate` - the control-variate estimator E[f_h] = mean(f_h samples) - beta *
    (mean(f_l at those points) - mean(f_l over all sampled points)), with beta fitted by least
    squares from the paired samples (this is MFMC/regression control variates).
  * `closed_form_allocation` - the optimal split of a cost budget between the fidelities,
    n_l/n_h = sqrt(rho^2/(1-rho^2) * c_h/c_l), checked against a brute-force minimisation of the
    predicted variance over the whole feasible allocation grid, plus the adaptive consequence:
    a surrogate with little correlation earns almost no budget.
  * `orchestrate_search` - the optimisation use: screen a fine candidate grid with the cheap
    model, then spend the remaining budget on high-fidelity refinement around the winner, and
    compare against spending the same budget uniformly on the high fidelity.

Simplified versus production: the sample design is a deterministic quadrature rather than Monte
Carlo, the fidelity gap is a fixed smooth bias, and only two fidelities and one worker are
modelled.  Verified below: truth-reference convergence, near-optimal allocation against the
grid search, exact budget accounting, the estimator beating the high-fidelity-only estimator at
equal cost, the fitted beta against the theoretical optimum, the weak-surrogate negative
control, the orchestrated search beating the uniform baseline, the exact-budget boundary, and
the reject-below-minimum-budget guard.
"""

import math

C_LOW = 1.0
C_HIGH = 10.0
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


class Rng:
    """splitmix64 - used only for the design shifts of the error ensemble, never for the model."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def unit(self):
        return (self.below(1 << 20) + 0.5) / (1 << 20)

    def below(self, n):
        return self.next_u64() % n


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def f_high(x):
    """Expensive truth: sin(2x) + 0.1x^2 on [0, 1]."""
    return math.sin(2.0 * x) + 0.1 * x * x


def bias(x):
    """The smooth discrepancy between the fidelities (the surrogate's systematic error)."""
    return 0.06 * math.sin(3.0 * x) + 0.02


def f_low(x):
    """Cheap surrogate: the truth shifted by the bias - positively correlated, as required."""
    return f_high(x) - bias(x)


def f_low_weak(x):
    """A deliberately poor surrogate: a high-frequency term destroys the correlation."""
    return f_high(x) + 2.0 * math.sin(11.0 * x)


def van_der_corput(index, base=2):
    """Point k of the Halton/van-der-Corput sequence - nested, so prefixes stay a design."""
    value, factor, k = 0.0, 1.0 / base, index
    while k > 0:
        value += factor * (k % base)
        k //= base
        factor /= base
    return value


def design(count, offset=0.0):
    """Nested van-der-Corput design; `offset` shifts the whole set (used for ensembles)."""
    return [(van_der_corput(i + 1) + offset) % 1.0 for i in range(count)]


def quadrature(function, count):
    return sum(function((i + 0.5) / count) for i in range(count)) / count


def moments(function, count=4096):
    """(E[f], E[g], Var f, Var g, rho) of the two fidelities on a fine midpoint grid."""
    xs = [(i + 0.5) / count for i in range(count)]
    left = [f_high(x) for x in xs]
    right = [function(x) for x in xs]
    mean_l = sum(left) / count
    mean_r = sum(right) / count
    var_l = sum((v - mean_l) ** 2 for v in left) / count
    var_r = sum((v - mean_r) ** 2 for v in right) / count
    cov = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right)) / count
    return mean_l, mean_r, var_l, var_r, cov / math.sqrt(var_l * var_r)


def mf_estimate(function, n_high, n_low, offset=0.0):
    """Control-variate estimate of E[f_high] under a fidelity split of the budget."""
    check(n_high >= 1, "at least one high-fidelity sample is required")
    n_low = max(n_low, n_high)              # the paired design points are always included
    points = design(n_low, offset)
    high = [f_high(x) for x in points[:n_high]]
    low = [function(x) for x in points]
    mean_high = sum(high) / n_high
    mean_low_paired = sum(low[:n_high]) / n_high
    mean_low_all = sum(low) / n_low
    if n_high < 2:
        return mean_high, 0.0
    var_low = sum((v - mean_low_paired) ** 2 for v in low[:n_high]) / (n_high - 1)
    cov = sum((h - mean_high) * (l - mean_low_paired)
              for h, l in zip(high, low[:n_high])) / (n_high - 1)
    beta = cov / var_low if var_low > 0 else 0.0
    return mean_high - beta * (mean_low_paired - mean_low_all), beta


def predicted_variance(var_high, rho, n_high, n_low):
    """Variance of the control-variate estimator with the optimal beta."""
    paired = var_high * (1.0 - rho * rho) / n_high
    if n_low <= 0:
        return var_high / n_high
    return paired + var_high * rho * rho / n_low


def closed_form_allocation(budget, rho, min_high=4):
    """Optimal (n_low, n_high) for a cost budget; n_low/n_high = sqrt(rho^2/(1-rho^2)*c_h/c_l).
    The continuous optimum assumes beta is known; at least `min_high` paired samples are kept so
    that the least-squares beta of the estimator can actually be fitted."""
    affordable = int(budget // C_HIGH)
    if affordable < 1:
        raise ValueError("budget %.1f cannot afford a single high-fidelity sample" % budget)
    if rho <= 0.0:
        return 0, affordable
    ratio = math.sqrt(rho * rho / (1.0 - rho * rho) * C_HIGH / C_LOW)
    n_high = max(min(affordable, min_high), int(budget // (C_HIGH + ratio * C_LOW)))
    n_low = int((budget - n_high * C_HIGH) // C_LOW)
    return n_low, n_high


def grid_allocation(budget, rho, var_high=1.0, min_high=4):
    """Brute-force oracle: the allocation grid point with the smallest predicted variance."""
    best = None
    for n_high in range(min_high, int(budget // C_HIGH) + 1):
        n_low = int((budget - n_high * C_HIGH) // C_LOW)
        variance = predicted_variance(var_high, rho, n_high, n_low)
        if best is None or variance < best[0]:
            best = (variance, n_low, n_high)
    return best


def uniform_search(count):
    """Baseline: spend the whole budget on evenly spread high-fidelity evaluations."""
    return max(f_high((i + 0.5) / count) for i in range(count))


def orchestrate_search(screen_points, refine_steps, function):
    """Screen with the cheap model, then refine around the winner with the expensive one."""
    grid = [(i + 0.5) / screen_points for i in range(screen_points)]
    scored = sorted(((function(x), x) for x in grid), reverse=True)
    winner = scored[0][1]
    step = 1.0 / screen_points
    best = (f_high(winner), winner)
    for k in range(1, refine_steps + 1):
        for x in (winner - step * k / 2.0, winner + step * k / 2.0):
            if 0.0 <= x <= 1.0:
                value = f_high(x)
                if value > best[0]:
                    best = (value, x)
    high_cost = (refine_steps * 2 + 1) * C_HIGH + screen_points * C_LOW
    return best, high_cost, winner


def main():
    truth = quadrature(f_high, 4096)
    coarse = quadrature(f_high, 2048)
    check(abs(truth - coarse) < 1e-4, "the quadrature oracle must be converged")
    check(0.7 < truth < 0.8, "the truth is in the expected range")

    # ---- 1. the surrogate is positively correlated, and the fitted beta tracks theory
    mean_h, mean_l, var_h, var_l, rho = moments(f_low)
    check(0.7 < rho < 1.0, "the surrogate must be strongly correlated, rho=%.4f" % rho)
    estimate, beta = mf_estimate(f_low, 16, 200)
    theory = rho * math.sqrt(var_h / var_l)
    check(abs(beta - theory) < 0.08,
          "the fitted beta %.4f must track rho*sigma_h/sigma_l = %.4f" % (beta, theory))
    check(abs(mean_h - truth) < 1e-3, "the grid moments must agree with the quadrature")

    # ---- 2. the closed-form allocation is near the brute-force optimum
    for budget in (60.0, 150.0, 400.0):
        n_low, n_high = closed_form_allocation(budget, rho)
        check(n_high * C_HIGH + n_low * C_LOW <= budget + 1e-9, "the budget must be respected")
        check(n_high >= 4, "enough paired samples must be paid for to fit beta")
        chosen = predicted_variance(var_h, rho, n_high, n_low)
        best_var, best_low, best_high = grid_allocation(budget, rho, var_h)
        check(best_var <= chosen,
              "no allocation may beat the chosen one at budget %.0f (%.6f vs %.6f)"
              % (budget, best_var, chosen))
        check(chosen <= best_var * 1.10,
              "the closed form must be within 10%% of the grid optimum at budget %.0f" % budget)
        check(chosen <= predicted_variance(var_h, rho, int(budget // C_HIGH), 0),
              "the split must beat spending the whole budget on the high fidelity")
    check(closed_form_allocation(150.0, rho)[0] > closed_form_allocation(150.0, rho)[1],
          "a strongly correlated surrogate earns more cheap samples than expensive ones")

    # ---- 3. multi-fidelity beats high-fidelity-only at the same cost, over a design ensemble
    for budget in (100.0, 400.0):
        n_low, n_high = closed_form_allocation(budget, rho)
        n_only = int(budget // C_HIGH)
        rng = Rng(906)
        sq_mf = sq_plain = 0.0
        for _ in range(64):
            offset = rng.unit()
            mf_value, _ = mf_estimate(f_low, n_high, n_low, offset)
            plain = sum(f_high(x) for x in design(n_only, offset)) / n_only
            sq_mf += (mf_value - truth) ** 2
            sq_plain += (plain - truth) ** 2
        check(sq_mf < sq_plain * 0.5,
              "the multi-fidelity estimator must halve the squared error at budget %.0f "
              "(%.6f vs %.6f)" % (budget, sq_mf / 64, sq_plain / 64))
        check(n_only >= n_high, "the baseline buys more, but less useful, samples")
        if budget == 100.0:
            ensemble = (sq_mf, sq_plain)

    # ---- 4. adaptive consequence: a poor surrogate earns relatively less budget
    _, _, _, _, weak_rho = moments(f_low_weak)
    check(-0.6 < weak_rho < 0.5, "the weak surrogate must have a small correlation: %.3f" % weak_rho)
    weak_low, weak_high = closed_form_allocation(150.0, max(weak_rho, 0.0))
    strong_low, strong_high = closed_form_allocation(150.0, rho)
    check(weak_low * strong_high < strong_low * weak_high,
          "the weak surrogate must earn fewer cheap samples per expensive one")
    silent_low, silent_high = closed_form_allocation(150.0, -0.5)
    check(silent_low == 0 and silent_high == 15,
          "a negatively correlated surrogate must be dropped entirely")
    check(predicted_variance(var_h, rho, strong_high, strong_low)
          < predicted_variance(var_h, max(weak_rho, 0.0), strong_high, strong_low),
          "at the better surrogate's own split the predicted variance must be lower")
    rng = Rng(1234)
    sq_strong = sq_weak = 0.0
    for _ in range(32):
        offset = rng.unit()
        sq_strong += (mf_estimate(f_low, strong_high, strong_low, offset)[0] - truth) ** 2
        sq_weak += (mf_estimate(f_low_weak, strong_high, strong_low, offset)[0] - truth) ** 2
    check(sq_strong < sq_weak,
          "the well-correlated surrogate must win on the ensemble (%.6f vs %.6f)"
          % (sq_strong / 32, sq_weak / 32))

    # ---- 5. orchestration: screen cheaply, refine expensively
    orchestrated, cost, winner = orchestrate_search(64, 2, f_low)
    uniform = uniform_search(11)
    check(cost <= 114.0, "the orchestrated search must stay inside the 114 unit budget")
    check(orchestrated[0] > uniform,
          "the orchestrated search %.5f must beat the uniform baseline %.5f"
          % (orchestrated[0], uniform))
    grid_best = max(f_high((i + 0.5) / 64) for i in range(64))
    check(orchestrated[0] <= grid_best + 1e-9, "refinement cannot invent a better optimum")
    check(abs(winner - 0.8359) < 0.06, "the screen must localise the peak region")

    # ---- 6. boundaries and determinism
    solo, solo_beta = mf_estimate(f_low, 1, 40)
    check(abs(solo - f_high(design(1)[0])) < 1e-12,
          "with a single high-fidelity sample the estimator degenerates to that sample")
    try:
        closed_form_allocation(5.0, rho)
        raise AssertionError("a budget below one high-fidelity sample must be rejected")
    except ValueError as error:
        check("cannot afford" in str(error), "the rejection must explain itself")
    for _ in range(2):
        check(mf_estimate(f_low, 8, 64) == mf_estimate(f_low, 8, 64), "estimates must be repeatable")
    check(design(8) == design(8) and design(4) == design(8)[:4],
          "the sample design must be nested and repeatable")

    print("truth %.6f, rho %.4f, variance reduction at budget 150: %.3fx"
          % (truth, rho,
             (var_h / int(150.0 // C_HIGH)) / predicted_variance(var_h, rho, 6, 90)))
    print("orchestrated %.5f at cost %.0f vs uniform %.5f at cost 110"
          % (orchestrated[0], cost, uniform))
    print("allocation at 150: n_low=%d n_high=%d" % closed_form_allocation(150.0, rho))
    print("PASS")


if __name__ == "__main__":
    main()
