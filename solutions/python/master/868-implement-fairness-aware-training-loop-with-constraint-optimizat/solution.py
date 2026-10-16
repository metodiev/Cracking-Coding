"""Task 868 - Implement fairness-aware training loop with constraint optimization (demographic parity).  (Tier: Master, CTCI Beyond - Machine Learning & AI)

The training loop minimises the regularised logistic risk

    L(w, b) = mean_i log(1 + exp(-(2 y_i - 1) (w . x_i + b)))

subject to the demographic-parity (DP) constraint

    d(w, b) = mean_{i in group A} sigmoid(w . x_i + b)
            - mean_{i in group B} sigmoid(w . x_i + b),        |d| <= eps.

The constrained problem is solved by its Lagrangian dual with a reduction (exponentiated
gradient) outer loop.  With multipliers theta_1, theta_2 >= 0 the inner problem is

    F(w, b) = L + theta_1 (d - eps) + theta_2 (-d - eps),

which is smooth and convex, so a single gradient-descent sweep with an explicitly computed
step size 1/Lip(F) minimises it.  The step size uses the exact Lipschitz bound

    Lip(grad F) <= 0.25 * (1 + |theta_1 - theta_2|) * max_i ||(x_i, 1)||^2

because the logistic Hessian is bounded by 0.25 * ||(x,1)||^2 and every DP derivative is an
average of p (1 - p) (x, 1) terms.  The multipliers follow the standard multiplicative
(exponentiated-gradient) update theta_k <- theta_k * exp(eta * g_k) with g = (d - eps,
-d - eps), floored at 1e-12 and capped, each block warm-started from the previous block.

Invariants verified below
  * safety - the returned model satisfies |d| <= eps (the trainer keeps the best *feasible*
    iterate it saw, so the constraint holds even though the raw dual iterates oscillate);
  * optimality - the returned loss is no worse than the best point of an exhaustive
    (w, b) grid that an independent brute-force oracle searches over the same feasible set;
  * liveness - the dual drive makes a previously violating iterate feasible, i.e. the
    violation sequence reaches zero within the iteration budget;
  * monotonicity of the efficiency/fairness trade-off (loss non-increasing in eps);
  * gradient correctness against central finite differences; determinism (bit-identical
    repeats); rejection of invalid inputs.

Simplified versus a production fairness trainer: a hand-written deterministic Gaussian
feature generator instead of a real dataset, full-batch gradient descent instead of SGD
with mini-batches, a fixed iteration budget instead of a duality-gap stopping rule, and
demographic parity on *mean predicted probability* (a differentiable surrogate for
positive-rate parity) so that the dual is differentiable.  Nothing is pre-computed: every
rate, gradient and multiplier is computed by the loops below.
"""
import math

FEASIBILITY_TOL = 1e-9
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------- deterministic PRNG
def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    """splitmix64 stream; the only source of randomness in this file."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        self.state, raw = splitmix64(self.state)
        return (raw >> 11) * (1.0 / float(1 << 53))

    def gauss(self):
        u = self.uniform()
        u = u if u > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * self.uniform())


# ------------------------------------------------------------------- model helpers
def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def make_dataset(seed, group_shifts=(0.7, -0.7), per_group=30, slope=1.6, spread=1.0,
                 extra_features=0):
    """Rows are (features, label, group).  P(y = 1 | x) = sigmoid(slope * x_0) exactly."""
    rng = Rng(seed)
    rows = []
    for group, shift in enumerate(group_shifts):
        for _ in range(per_group):
            features = [shift + spread * rng.gauss()]
            for _ in range(extra_features):
                features.append(rng.gauss())
            u = rng.uniform()
            u = u if u > 1e-15 else 1e-15
            label = 1 if slope * features[0] > math.log(u / (1.0 - u)) else 0
            rows.append((features, label, group))
    return rows


def validate(rows, eps):
    if not rows:
        raise ValueError("empty dataset")
    if not (eps >= 0.0) or math.isnan(eps):
        raise ValueError("eps must be finite and non-negative")
    dim = len(rows[0][0])
    if any(len(features) != dim for features, _label, _group in rows):
        raise ValueError("ragged feature vectors")
    for _features, _label, group in rows:
        if group not in (0, 1):
            raise ValueError("unknown demographic group")


def lipschitz_bound(rows, theta):
    """Exact bound on ||grad^2 F||_2 used to pick the inner step size."""
    maximum = 0.0
    for features, _label, _group in rows:
        norm_sq = 1.0
        for value in features:
            norm_sq += value * value
        if norm_sq > maximum:
            maximum = norm_sq
    return 0.25 * (1.0 + abs(theta[0] - theta[1])) * maximum


def loss_and_grad(rows, params):
    """Mean logistic loss and its gradient wrt [weights..., bias]."""
    dim = len(params)
    bias = params[-1]
    loss = 0.0
    grad = [0.0] * dim
    for features, label, _group in rows:
        z = bias
        for k in range(dim - 1):
            z += params[k] * features[k]
        if label == 1:
            loss += math.log1p(math.exp(-z))
        else:
            loss += math.log1p(math.exp(z))
        residual = sigmoid(z) - label
        for k in range(dim - 1):
            grad[k] += residual * features[k]
        grad[-1] += residual
    inv = 1.0 / len(rows)
    return loss * inv, [value * inv for value in grad]


def dp_and_grad(rows, params):
    """Signed demographic-parity gap, its gradient and the two group positive rates.

    With a single demographic group present the gap is defined as 0 (nothing to compare),
    which makes the fairness constraint vacuous instead of raising a division by zero.
    """
    dim = len(params)
    bias = params[-1]
    counts = {}
    sums = {}
    grads = {}
    for features, _label, group in rows:
        z = bias
        for k in range(dim - 1):
            z += params[k] * features[k]
        p = sigmoid(z)
        counts[group] = counts.get(group, 0) + 1
        sums[group] = sums.get(group, 0.0) + p
        row_grad = grads.setdefault(group, [0.0] * dim)
        slope = p * (1.0 - p)
        for k in range(dim - 1):
            row_grad[k] += slope * features[k]
        row_grad[-1] += slope
    if len(counts) < 2:
        only = next(iter(counts))
        return 0.0, [0.0] * dim, {only: sums[only] / counts[only]}, counts
    rates = {}
    for group, count in counts.items():
        rates[group] = sums[group] / count
        inv = 1.0 / count
        for k in range(dim):
            grads[group][k] *= inv
    gap = rates[0] - rates[1]
    grad = [grads[0][k] - grads[1][k] for k in range(dim)]
    return gap, grad, rates, counts


def objective(rows, params, theta, eps):
    """The inner (Lagrangian) objective F; used by the gradient check."""
    loss, _grad = loss_and_grad(rows, params)
    gap, _grad_gap, _rates, _counts = dp_and_grad(rows, params)
    return loss + theta[0] * (gap - eps) + theta[1] * (-gap - eps)


def inner_solve(rows, params, theta, eps, steps):
    """Fixed-step gradient descent on F, warm-started from `params`.

    Returns the final iterate and the best *feasible* iterate seen (loss, params).
    """
    bound = lipschitz_bound(rows, theta)
    lr = 0.9 / bound if bound > 1e-12 else 4.0
    current = list(params)
    best = None
    for _ in range(steps):
        loss, grad_loss = loss_and_grad(rows, current)
        gap, grad_gap, _rates, _counts = dp_and_grad(rows, current)
        if abs(gap) <= eps + FEASIBILITY_TOL and (best is None or loss < best[0]):
            best = (loss, list(current))
        weight = theta[0] - theta[1]
        for k in range(len(current)):
            current[k] -= lr * (grad_loss[k] + weight * grad_gap[k])
    return current, best


def train_fair(rows, eps, outer_steps=60, inner_steps=80, dual_rate=1.0, theta_cap=25.0):
    """Reduction-style constrained training loop; returns a dict of metrics."""
    validate(rows, eps)
    params = [0.0] * (len(rows[0][0]) + 1)
    # Equal positive multipliers make the penalty term constant (theta_1 - theta_2 == 0), so
    # the first inner solve is the plain unconstrained risk while still giving the
    # multiplicative rule a non-degenerate starting point.
    theta = [1.0, 1.0]
    best = None
    violations = []
    theta_peak = 0.0
    for _ in range(outer_steps):
        params, candidate = inner_solve(rows, params, theta, eps, inner_steps)
        if candidate is not None and (best is None or candidate[0] < best[0]):
            best = candidate
        gap, _grad, _rates, _counts = dp_and_grad(rows, params)
        violations.append(max(gap - eps, -gap - eps, 0.0))
        theta[0] = min(theta_cap, max(1e-12, theta[0] * math.exp(dual_rate * (gap - eps))))
        theta[1] = min(theta_cap, max(1e-12, theta[1] * math.exp(dual_rate * (-gap - eps))))
        theta_peak = max(theta_peak, theta[0], theta[1])
    if best is None:  # pragma: no cover - the all-zero start is always feasible for eps >= 0
        best = (loss_and_grad(rows, params)[0], params)
    chosen_loss, chosen = best
    gap, _grad, rates, counts = dp_and_grad(rows, chosen)
    return {
        "params": chosen,
        "weights": chosen[:-1],
        "bias": chosen[-1],
        "loss": chosen_loss,
        "gap": gap,
        "rates": rates,
        "counts": counts,
        "theta": theta,
        "theta_peak": theta_peak,
        "violations": violations,
        "last_params": params,
        "feasible_found": candidate is not None,
    }


# ------------------------------------------------------------------ brute-force oracle
def grid_search(rows, eps, limit=4.0, steps=61, refine_rounds=6, refine_steps=11):
    """Exhaustive search for the loss-minimising feasible (w, b), with box refinement.

    The coarse sweep is an independent brute-force oracle over the same feasible set; the
    refinement rounds shrink the box around the incumbent so that the oracle is accurate
    enough to compare the solver's loss against with a small absolute tolerance.
    """
    best = None
    feasible = 0
    span = 2.0 * limit / (steps - 1)
    for i in range(steps):
        w = -limit + span * i
        for j in range(steps):
            b = -limit + span * j
            params = [w, b]
            gap, _grad, _rates, _counts = dp_and_grad(rows, params)
            if abs(gap) <= eps + FEASIBILITY_TOL:
                feasible += 1
                loss = loss_and_grad(rows, params)[0]
                if best is None or loss < best[0]:
                    best = (loss, params)
    if best is None:
        return None, 0
    centre = list(best[1])
    radius = span
    for _ in range(refine_rounds):
        radius *= 0.5
        for i in range(refine_steps):
            for j in range(refine_steps):
                params = [centre[0] - radius + 2.0 * radius * i / (refine_steps - 1),
                          centre[1] - radius + 2.0 * radius * j / (refine_steps - 1)]
                gap, _grad, _rates, _counts = dp_and_grad(rows, params)
                if abs(gap) > eps + FEASIBILITY_TOL:
                    continue
                feasible += 1
                loss = loss_and_grad(rows, params)[0]
                if loss < best[0]:
                    best = (loss, list(params))
        centre = list(best[1])
    return best, feasible


def main():
    rows = make_dataset(seed=20260868)
    labels = sum(label for _f, label, _g in rows)
    check(len(rows) == 60 and 0 < labels < 60, "generator must produce two labels")
    check({group for _f, _l, group in rows} == {0, 1}, "generator must produce two groups")

    # ---- the unconstrained optimum is unfair, so the constraint is binding ----
    unconstrained = train_fair(rows, eps=10.0)
    check(unconstrained["rates"][0] > unconstrained["rates"][1],
          "unconstrained model must favour the group with the larger feature shift")
    unconstrained_gap = abs(unconstrained["gap"])
    check(unconstrained_gap > 0.2,
          f"unconstrained solution should be strongly unfair, gap={unconstrained_gap:.4f}")

    eps_tight = 0.5 * unconstrained_gap
    fair = train_fair(rows, eps=eps_tight)

    # safety: the returned model is feasible even though raw dual iterates oscillate
    check(abs(fair["gap"]) <= eps_tight + 1e-9,
          f"demographic parity violated: gap={fair['gap']:.6f} eps={eps_tight:.6f}")
    check(abs(fair["rates"][0] - fair["rates"][1])
          < abs(unconstrained["rates"][0] - unconstrained["rates"][1]),
          "constrained model must shrink the between-group prediction gap")
    check(abs(fair["gap"]) > 0.7 * eps_tight,
          f"solver overshoots: gap={abs(fair['gap']):.6f} far below eps={eps_tight:.6f}")

    # efficiency cost of fairness is real but bounded
    delta = fair["loss"] - unconstrained["loss"]
    check(delta > 1e-9, f"constrained loss must exceed the unconstrained loss ({delta:.6f})")
    check(delta < 0.5, f"fairness cost unexpectedly large: delta={delta:.6f}")

    # liveness: the dual drive reaches feasibility during the run
    check(max(fair["violations"]) > 1e-3,
          "the raw dual iterates should violate the constraint at some point")
    check(min(fair["violations"]) <= 1e-3,
          f"dual updates never produced a feasible iterate: {min(fair['violations']):.3e}")
    half = len(fair["violations"]) // 2
    check(min(fair["violations"][half:]) <= 1e-3,
          "feasibility was not reached in the second half of the dual iterations")
    check(min(fair["violations"]) < 0.01 * max(fair["violations"]),
          "the dual drive must reduce the violation by at least two orders of magnitude")

    # optimality against an independent exhaustive oracle on the same feasible set
    grid_best, feasible_points = grid_search(rows, eps_tight)
    check(grid_best is not None and feasible_points > 100, "grid oracle found no feasible point")
    check(grid_best[0] >= unconstrained["loss"] - 1e-6,
          "grid oracle should not beat the unconstrained optimum")
    check(fair["loss"] <= grid_best[0] + 5e-3,
          f"loss {fair['loss']:.6f} worse than grid optimum {grid_best[0]:.6f}")

    # the dual variable for the binding direction grows large and the other one stays small
    check(fair["theta_peak"] > 1.0,
          f"binding constraint must grow a non-trivial multiplier, peak={fair['theta_peak']}")
    check(min(fair["theta"]) < fair["theta_peak"],
          "only the violated direction may accumulate dual weight")

    # monotone efficiency/fairness trade-off
    losses = []
    for eps in (0.02, 0.15, 0.40, 0.90):
        run = train_fair(rows, eps=eps, outer_steps=30, inner_steps=60)
        check(abs(run["gap"]) <= eps + 1e-9, f"eps={eps}: parity not satisfied")
        losses.append(run["loss"])
    for previous, current in zip(losses, losses[1:]):
        check(current <= previous + 1e-9,
              f"loss must be non-increasing in eps: {losses}")

    # determinism
    repeat = train_fair(rows, eps=eps_tight)
    check(repeat["params"] == fair["params"] and repeat["loss"] == fair["loss"]
          and repeat["gap"] == fair["gap"], "training must be deterministic")

    # analytic gradient against central finite differences
    params = [0.4, -0.2]
    theta = [0.7, 0.1]
    _loss, grad = loss_and_grad(rows, params)
    gap_deriv, grad_gap, _rates, _counts = dp_and_grad(rows, params)
    analytic = [grad[k] + (theta[0] - theta[1]) * grad_gap[k] for k in range(2)]
    step = 1e-6
    for k in range(2):
        plus = list(params)
        minus = list(params)
        plus[k] += step
        minus[k] -= step
        numeric = (objective(rows, plus, theta, eps_tight)
                   - objective(rows, minus, theta, eps_tight)) / (2.0 * step)
        check(abs(numeric - analytic[k]) <= 1e-5 * max(1.0, abs(analytic[k])),
              f"gradient mismatch at {k}: analytic={analytic[k]:.8f} numeric={numeric:.8f}")
    check(abs(gap_deriv - dp_and_grad(rows, params)[0]) == 0.0, "gap must be stable")

    # multi-feature case: the solver is dimension generic
    wide = make_dataset(seed=7, extra_features=2, per_group=25)
    wide_run = train_fair(wide, eps=0.15)
    wide_plain = train_fair(wide, eps=10.0)
    check(abs(wide_run["gap"]) <= 0.15 + 1e-9, "2-D case: parity not satisfied")
    check(wide_run["loss"] >= wide_plain["loss"] - 1e-9,
          "2-D case: constraint must not lower the loss")

    # single group: the constraint is vacuous, so training reduces to plain ERM
    single = [(features, label, 0) for features, label, _group in rows]
    single_fair = train_fair(single, eps=0.0)
    single_plain = train_fair(single, eps=10.0)
    check(single_fair["gap"] == 0.0, "single-group gap must be zero")
    check(abs(single_fair["loss"] - single_plain["loss"]) <= 1e-12,
          "single group: eps must be irrelevant")

    # invalid inputs are rejected
    for bad_rows, bad_eps, why in ((rows, -0.5, "negative eps"), ([], 0.1, "empty dataset"),
                                   ([(features, label, 3) for features, label, _g in rows],
                                    0.1, "unknown group")):
        try:
            train_fair(bad_rows, eps=bad_eps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    print(f"rows={len(rows)} eps_tight={eps_tight:.4f}")
    print(f"unconstrained loss={unconstrained['loss']:.5f} gap={unconstrained_gap:.5f}")
    print(f"fair loss={fair['loss']:.5f} gap={fair['gap']:.5f} theta_peak={fair['theta_peak']:.3f}")
    print(f"grid optimum={grid_best[0]:.5f} over {feasible_points} feasible evaluations")
    print("PASS")


if __name__ == "__main__":
    main()
