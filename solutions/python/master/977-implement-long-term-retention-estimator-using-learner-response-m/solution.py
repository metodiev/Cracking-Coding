"""Task 977 - Implement long-term retention estimator using learner response models.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A half-life regression style response model: after n successful reviews the memory half-life is
h = 2**(theta0 + theta1 * n) days, and the probability of a later recall after delta days is the
exponential forgetting curve p = 2**(-delta / h).  The parameters are fitted to observed review outcomes
by minimising the L2-regularised negative log-likelihood, the analytic gradient of the loss is checked
against central finite differences, and the fitted model is judged by calibration (does the predicted
recall match the observed rate per prediction bin?) and by log-loss against a base-rate baseline on a
held-out split.

Simplified versus a production estimator (Duolingo's HLR): two features instead of the full feature vector,
one global model instead of per-learner models, and a deterministic coordinate search on a shrinking grid
rather than stochastic gradient descent - chosen so the fit is reproducible and the loss is non-increasing
by construction.  Verified: recovery of the planted parameters, the analytic gradient, the curve's
boundary and monotonicity properties, a bisection oracle for the half-life, bin calibration, held-out
log-loss versus the baseline and versus a label-shuffled fit, empty and fresh-item boundaries, and
reproducibility.
"""
import math

PARAMETERS = {"examples": 900, "split": 640, "spread": 22}
TRUE_THETA = (2.0, 0.5)      # planted model: h = 2**(2 + 0.5 * n) days
START_THETA = (1.0, 0.2)
L2 = 1e-4
MASK64 = (1 << 64) - 1
LN2 = math.log(2.0)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - a fixed seed makes the simulated review history reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def half_life(theta, successes):
    """Memory half-life in days: grows with the number of successful reviews."""
    return 2.0 ** (theta[0] + theta[1] * successes)


def predict(theta, successes, days):
    """Probability that the learner still recalls the item `days` after the last review."""
    return 2.0 ** (-days / half_life(theta, successes))


def simulate(seed=11, count=PARAMETERS["examples"], theta=TRUE_THETA):
    """(successes, days since review, recalled) drawn from the planted model."""
    rng = Rng(seed)
    examples = []
    for _ in range(count):
        successes = rng.below(5)
        days = 1 + rng.below(30)
        recalled = rng.below(1000) < int(round(1000 * predict(theta, successes, days)))
        examples.append((successes, days, int(recalled)))
    return examples


def negative_log_likelihood(theta, examples, l2=L2):
    total = 0.5 * l2 * sum(value * value for value in theta)
    for successes, days, recalled in examples:
        probability = predict(theta, successes, days)
        probability = min(max(probability, 1e-12), 1 - 1e-12)
        total -= math.log(probability) if recalled else math.log(1 - probability)
    return total


def gradient(theta, examples, l2=L2):
    """Analytic gradient: dL/dtheta_j = -(y/p - (1-y)/(1-p)) * p * z * ln2 * x_j with z = ln2*delta/h."""
    grad = [l2 * theta[0], l2 * theta[1]]
    for successes, days, recalled in examples:
        probability = min(max(predict(theta, successes, days), 1e-12), 1 - 1e-12)
        z = LN2 * days / half_life(theta, successes)
        pull = probability * z * LN2
        weight = recalled / probability - (1 - recalled) / (1 - probability)
        grad[0] -= weight * pull
        grad[1] -= weight * pull * successes
    return grad


def numeric_gradient(theta, examples, l2=L2, epsilon=1e-6):
    """Independent oracle for the gradient: central differences of the loss."""
    out = []
    for index in range(len(theta)):
        up, down = list(theta), list(theta)
        up[index] += epsilon
        down[index] -= epsilon
        out.append((negative_log_likelihood(up, examples, l2)
                    - negative_log_likelihood(down, examples, l2)) / (2 * epsilon))
    return out


def fit(examples, steps=44, l2=L2):
    """Coordinate search on a shrinking grid: deterministic and loss-decreasing by construction."""
    theta, sizes = list(START_THETA), [1.0, 0.5]
    best = negative_log_likelihood(theta, examples, l2)
    history = [best]
    for _ in range(steps):
        improved = False
        for index in range(len(theta)):
            for direction in (-1.0, 1.0):
                candidate = list(theta)
                candidate[index] += direction * sizes[index]
                value = negative_log_likelihood(candidate, examples, l2)
                if value < best - 1e-12:
                    theta, best, improved = candidate, value, True
        history.append(best)
        if not improved:
            sizes = [size / 2 for size in sizes]
    return theta, history


def bisect_days(theta, successes, target=0.5, low=0.0, high=200.0, rounds=80):
    """Independent oracle for the half-life: bisect the curve until the recall probability is 0.5."""
    for _ in range(rounds):
        middle = (low + high) / 2
        if predict(theta, successes, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def mean(values):
    return sum(values) / len(values) if values else 0.0


def log_loss(theta, examples):
    """Mean binary cross-entropy of the model on a set of reviews."""
    total = 0.0
    for successes, days, recalled in examples:
        probability = min(max(predict(theta, successes, days), 1e-12), 1 - 1e-12)
        total -= math.log(probability) if recalled else math.log(1 - probability)
    return total / len(examples) if examples else 0.0


def calibration(theta, examples, bins=4):
    """[(bin label, mean predicted recall, observed recall rate, count)] over the prediction range."""
    rows = sorted((predict(theta, successes, days), float(recalled))
                  for successes, days, recalled in examples)
    size = max(len(rows) // bins, 1)
    table = []
    for index in range(bins):
        chunk = rows[index * size:(index + 1) * size] if index < bins - 1 else rows[index * size:]
        if chunk:
            table.append((index + 1, mean([value for value, _y in chunk]), mean([y for _v, y in chunk]),
                          len(chunk)))
    return table


def shuffle_labels(examples, seed=3):
    """Break the association between features and outcome while keeping the base rate."""
    rng = Rng(seed)
    labels = [recalled for _n, _d, recalled in examples]
    for index in range(len(labels) - 1, 0, -1):
        other = rng.below(index + 1)
        labels[index], labels[other] = labels[other], labels[index]
    return [(successes, days, labels[index]) for index, (successes, days, _y) in enumerate(examples)]


def main():
    examples = simulate()
    train, held_out = examples[:PARAMETERS["split"]], examples[PARAMETERS["split"]:]
    # 1) the simulated history shows forgetting: the recall rate falls with the delay and rises with
    #    the number of successful reviews (the signal the fit has to pick up).
    early = [y for _n, days, y in examples if days <= 8]
    late = [y for _n, days, y in examples if days >= 22]
    few = [y for n, _days, y in examples if n <= 1]
    many = [y for n, _days, y in examples if n >= 3]
    check(len(examples) == PARAMETERS["examples"] and mean(early) > mean(late) + 0.2
          and mean(many) > mean(few) + 0.2,
          f"the simulated history has no signal: early={mean(early):.3f} late={mean(late):.3f}")
    # 2) the fit recovers the planted parameters, and its loss never increases along the way.
    theta, history = fit(train)
    check(all(history[i + 1] <= history[i] + 1e-12 for i in range(len(history) - 1))
          and history[-1] < history[0] and abs(theta[0] - TRUE_THETA[0]) < 0.15
          and abs(theta[1] - TRUE_THETA[1]) < 0.15 and theta[1] > 0,
          f"the fit did not recover the planted model: {theta} vs {TRUE_THETA}")
    # 3) the analytic gradient matches central differences at two very different parameter settings.
    for point in (START_THETA, theta):
        analytic, numeric = gradient(point, examples), numeric_gradient(point, examples)
        check(all(abs(left - right) <= 1e-6 + 1e-3 * abs(right) for left, right in zip(analytic, numeric)),
              f"the analytic gradient is wrong at {point}: {analytic} vs {numeric}")
    # 4) curve properties and boundaries: certainty right after a review, strict decay afterwards.
    check(predict(theta, 3, 0) == 1.0 and 0.0 < predict(theta, 3, 200) < 0.05
          and all(predict(theta, 3, delta) > predict(theta, 3, delta + 1) for delta in range(0, 40))
          and all(half_life(theta, n) < half_life(theta, n + 1) for n in range(6))
          and all(0.0 <= predict(theta, n, days) <= 1.0 for n in range(6) for days in range(0, 60, 7)),
          f"the forgetting curve is not well behaved: {[round(predict(theta, 3, d), 3) for d in range(0, 20, 4)]}")
    # 5) bisection oracle: the day where recall reaches 0.5 is the closed-form half-life.
    for successes in range(6):
        found = bisect_days(theta, successes)
        check(abs(found - half_life(theta, successes)) < 1e-6
              and abs(predict(theta, successes, found) - 0.5) < 1e-9,
              f"bisection disagrees for n={successes}: {found} vs {half_life(theta, successes)}")
    # 6) calibration on the training split: every prediction bin is close to the observed rate, and the
    #    overall mean prediction is close to the base rate.
    table = calibration(theta, train)
    check(len(table) == 4 and all(abs(predicted - observed) <= 0.1 for _b, predicted, observed, _c in table)
          and all(count >= 40 for _b, _p, _o, count in table)
          and abs(mean([predict(theta, n, days) for n, days, _y in train]) - mean([y for _n, _d, y in train]))
          < 0.03,
          f"the model is not calibrated: {[(b, round(p, 3), round(o, 3)) for b, p, o, _c in table]}")
    # 7) held-out comparison: beating the base-rate baseline, and losing that edge when the training
    #    labels are shuffled - so the fit really learned the shape of forgetting.
    base_rate = mean([y for _n, _d, y in train])
    baseline_loss = mean([-math.log(base_rate) if y else -math.log(1 - base_rate) for _n, _d, y in held_out])
    shuffled_theta, _shuffled_history = fit(shuffle_labels(train))
    check(log_loss(theta, held_out) < baseline_loss - 0.01
          and log_loss(theta, held_out) < log_loss(shuffled_theta, held_out) - 0.005,
          f"the fitted model does not beat the baselines: {log_loss(theta, held_out):.4f} "
          f"vs baseline {baseline_loss:.4f} vs shuffled {log_loss(shuffled_theta, held_out):.4f}")
    # 8) boundaries: with no data the regulariser pulls the parameters to zero (a fresh item is still
    #    certain right after a review), and the empty-data loss falls to the pure L2 term.
    empty_theta, empty_history = fit([])
    check(abs(empty_theta[0]) < 0.01 and abs(empty_theta[1]) < 0.01
          and all(empty_history[i + 1] <= empty_history[i] + 1e-15 for i in range(len(empty_history) - 1))
          and empty_history[-1] < empty_history[0] and abs(empty_history[-1]) < 1e-9
          and predict(empty_theta, 0, 0) == 1.0 and negative_log_likelihood(TRUE_THETA, []) > 0.0
          and gradient(TRUE_THETA, []) == [L2 * TRUE_THETA[0], L2 * TRUE_THETA[1]]
          and log_loss(theta, []) == 0.0 and bisect_days(theta, 0) < bisect_days(theta, 4),
          "the empty and fresh-item boundaries are wrong")
    # 9) reproducibility: the simulated history, the fit and its loss history are byte-identical.
    check(simulate() == examples and fit(train) == (theta, history) and simulate(seed=11) == simulate()
          and numeric_gradient(START_THETA, examples) == numeric_gradient(START_THETA, examples)
          and calibration(theta, train) == table,
          "the estimator must be reproducible")
    print(f"examples={len(examples)} train={len(train)} held_out={len(held_out)} "
          f"fitted_theta=({theta[0]:.3f}, {theta[1]:.3f}) planted={TRUE_THETA} "
          f"loss={history[0]:.4f}->{history[-1]:.4f}")
    print(f"held_out_logloss={log_loss(theta, held_out):.4f} baseline={baseline_loss:.4f} "
          f"shuffled={log_loss(shuffled_theta, held_out):.4f} half_lives="
          f"{[round(half_life(theta, n), 2) for n in range(5)]}")
    print("PASS")


if __name__ == "__main__":
    main()
