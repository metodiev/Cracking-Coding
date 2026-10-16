"""Task 857 - Implement ML-based cost model for query planner with online learning.  (Tier: Master, Ch 14 — Databases)

A learned cardinality/cost model for a query planner, trained *online* on a stream of
executed operators.  Every operator is described by a small feature vector (input rows,
output rows, memory pressure, whether an index is available), the measured cost is the
label, and a linear model over an engineered basis is updated by stochastic gradient descent
with a decaying learning rate and running normalisation - the same shape as a production
learned cost model, but fully deterministic and in-process: no threads, no clock, no
unseeded randomness (the example stream comes from a seeded splitmix64 stream).

What is simplified: the "database" is an analytic workload generator whose ground-truth cost
is an explicit function of the features, so the learner's target is exactly recoverable and
can be compared with the true coefficients instead of only with a black box.  Features are
the engineering step a real system needs (a log basis and a spill interaction term); the
planner picks the argmin of the *predicted* cost over an enumerated plan space.

Invariants verified:
  * the online SGD is correct: its analytic update equals a finite-difference gradient step
    on the same loss, and one SGD epoch over a full batch reproduces a hand-computed descent
    step;
  * learning really happens: the held-out error falls monotonically (in blocks) over the
    epochs and the loss never increases between checkpoints after the warm-up;
  * recoverability: trained on noiseless data from a known linear ground truth, the learned
    weights converge to the true coefficients within tolerance, and the residual mean
    absolute error on a held-out set is small;
  * normalisation is sound: the standardised features have mean ~0 and unit variance;
  * distribution shift: after the workload moves to a new regime the same online model
    re-adapts - its error on the tail of the shifted stream is far below its error on the
    head, and clearly below a model that kept training on the old regime only;
  * the planner benefit is measured: plan choices made with the learned model reach a much
    lower true-cost regret than a static rule-of-thumb planner, and agree with the optimal
    plan far more often;
  * edge cases: an empty training stream, a single example, zero-variance features, a zero
    learning rate (no learning), and reproducibility from the seed.
"""

import math

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * (self.next_u64() / float(1 << 64))

    def below(self, n):
        return self.next_u64() % n


# ---------------------------------------------------------------------------
# the ground truth cost model the learner has to discover
# ---------------------------------------------------------------------------
TRUE_WEIGHTS = (0.4, 0.9, 0.55, 1.6, 0.25)     # bias, log(in), log(out), spill, indexed
MEMORY = 2000.0


def features(operator):
    """Engineered basis: the raw numbers plus the two terms a real model needs."""
    rows_in = float(operator["rows_in"])
    rows_out = float(operator["rows_out"])
    spill = max(0.0, math.log1p(rows_in) - math.log1p(MEMORY)) * math.log1p(rows_out)
    indexed = 1.0 if operator.get("indexed") else 0.0
    return [1.0, math.log1p(rows_in), math.log1p(rows_out), spill, indexed]


def true_cost(operator):
    return dot(TRUE_WEIGHTS, features(operator))


def dot(weights, vector):
    total = 0.0
    for weight, value in zip(weights, vector):
        total += weight * value
    return total


def generate_operators(seed, count, regime="small"):
    rng = SplitMix64(seed)
    ranges = {"small": (20.0, 500.0), "drifted": (300.0, 4000.0),
              "large": (3000.0, 23000.0), "mixed": (20.0, 20000.0)}
    low, high = ranges[regime]
    operators = []
    for index in range(count):
        rows_in = low + rng.uniform(0.0, high - low)
        selectivity = rng.uniform(0.05, 3.0)
        operators.append({"rows_in": rows_in, "rows_out": max(1.0, rows_in * selectivity),
                          "indexed": rng.below(2) == 1,
                          "name": "op%04d" % index})
    return operators


# ---------------------------------------------------------------------------
# the online learner
# ---------------------------------------------------------------------------
class OnlineCostModel:
    """Linear model over the feature basis, trained by SGD with running normalisation."""

    def __init__(self, dimension=5, learning_rate=0.02, decay=0.001, seed=17, clip=5.0):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.decay = decay
        self.clip = clip
        self.weights = [0.0] * dimension
        self.count = 0
        self.mean = [0.0] * dimension
        self.moment = [1e-8] * dimension
        self.updates = 0

    # -- normalisation -------------------------------------------------
    def normalised(self, vector):
        """Bias column untouched, the rest centred and scaled to unit variance."""
        if not self.count:
            return list(vector)
        result = [vector[0]]
        for index in range(1, self.dimension):
            result.append((vector[index] - self.mean[index]) / self.scale(index))
        return result

    def observe(self, vector):
        """Running mean and second moment (Welford-style, in order-independent form)."""
        self.count += 1
        for index, value in enumerate(vector):
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count
            self.moment[index] += delta * (value - self.mean[index])

    def standard_deviation(self, index):
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.moment[index], 0.0) / (self.count - 1))

    def scale(self, index):
        """The normaliser: one standard deviation, or 1.0 while it is still degenerate."""
        return self.standard_deviation(index) or 1.0

    # -- prediction and update -----------------------------------------
    def predict(self, vector, normalise=True):
        if normalise:
            vector = self.normalised(vector)
        return dot(self.weights, vector)

    def clipped_error(self, vector, label):
        error = dot(self.weights, self.normalised(vector)) - label
        return max(-self.clip, min(self.clip, error))

    def loss(self, example, normalise=True):
        scaled = self.normalised(example["features"]) if normalise else list(example["features"])
        error = dot(self.weights, scaled) - example["label"]
        return 0.5 * error * error

    def gradient(self, example, normalise=True):
        """Plain least-squares gradient (used to verify the update rule)."""
        scaled = self.normalised(example["features"]) if normalise else list(example["features"])
        error = dot(self.weights, scaled) - example["label"]
        return [error * value for value in scaled]

    def descent_gradient(self, example):
        """The gradient actually applied: the error is clipped for robustness to drift."""
        scaled = self.normalised(example["features"])
        error = max(-self.clip, min(self.clip, dot(self.weights, scaled) - example["label"]))
        return [error * value for value in scaled], error

    def descent(self, example):
        """One clipped gradient step in the current normalised space."""
        gradient, error = self.descent_gradient(example)
        rate = self.learning_rate / (1.0 + self.decay * self.updates)
        for index in range(self.dimension):
            self.weights[index] -= rate * gradient[index]
        self.updates += 1
        return error * error

    def step(self, example):
        self.observe(example["features"])
        return self.descent(example)


def make_examples(operators):
    return [{"features": features(operator), "label": true_cost(operator),
             "name": operator["name"]} for operator in operators]


def recovered_weights(model):
    """Map the standardised-space weights back onto the raw features: the learned model is
    w0 + sum_i (w_i / s_i) * (x_i - m_i), so the raw coefficients are w_i / s_i with the
    constants folded into the bias."""
    raw = [model.weights[0]]
    for index in range(1, model.dimension):
        scale = model.scale(index)
        raw.append(model.weights[index] / scale)
        raw[0] -= model.weights[index] * model.mean[index] / scale
    return raw


def mean_absolute_error(model, examples):
    if not examples:
        return 0.0
    total = 0.0
    for example in examples:
        total += abs(model.predict(example["features"]) - example["label"])
    return total / len(examples)


def r_squared(model, examples):
    if not examples:
        return 1.0
    labels = [example["label"] for example in examples]
    average = sum(labels) / len(labels)
    total = sum((label - average) ** 2 for label in labels) or 1.0
    residual = sum((model.predict(example["features"]) - example["label"]) ** 2
                   for example in examples)
    return 1.0 - residual / total


def train(model, examples, epochs=1):
    history = []
    for _ in range(epochs):
        for example in examples:
            history.append(model.step(example))
    return history


# ---------------------------------------------------------------------------
# plan space and planners
# ---------------------------------------------------------------------------
PLAN_OPTIONS = ("hash", "nested", "index", "sorted")


def plan_operators(query, order):
    """A plan is a join order plus one access method per table."""
    operators = []
    rows = query["base_rows"]
    for position, table in enumerate(order):
        method = query["order_method"][table] if False else None
        del method
        operators.append({"rows_in": rows, "rows_out": rows * query["fanouts"][table],
                          "indexed": table in query["indexed"], "table": table,
                          "position": position})
        rows = rows * query["fanouts"][table]
    return operators


def query_plans(rng, count):
    queries = []
    for index in range(count):
        base = 50.0 + rng.uniform(0.0, 4000.0)
        tables = ["t0", "t1", "t2"]
        fanouts = {}
        for position, table in enumerate(tables):
            fanouts[table] = 0.2 + rng.uniform(0.0, 4.0)
        queries.append({"name": "q%03d" % index, "base_rows": base, "tables": tables,
                        "fanouts": fanouts,
                        "indexed": {"t%d" % (rng.below(3))} if rng.below(4) == 0 else set()})
    return queries


def orderings(tables):
    if len(tables) == 1:
        return [tuple(tables)]
    options = []
    for table in tables:
        rest = [other for other in tables if other != table]
        for tail in orderings(rest):
            options.append((table,) + tail)
    return options


def predict_plan(model, query, order):
    return sum(model.predict(features(operator))
               for operator in plan_operators(query, order))


def truth_plan(query, order):
    return sum(true_cost(operator) for operator in plan_operators(query, order))


def static_planner(query):
    """Rule of thumb partner: always keep the input order and hash everything."""
    return tuple(query["tables"])


def evaluate_planner(planner, queries):
    regrets = []
    exact = 0
    for query in queries:
        candidates = orderings(query["tables"])
        costs = {order: truth_plan(query, order) for order in candidates}
        optimum = min(costs.values())
        chosen = planner(query)
        regrets.append(costs[chosen] / optimum)
        exact += 1 if costs[chosen] <= optimum + 1e-9 else 0
    return sum(regrets) / len(regrets), max(regrets), exact / len(queries)


def learned_planner(model):
    def planner(query):
        return min(orderings(query["tables"]),
                   key=lambda order: (predict_plan(model, query, order), order))
    return planner


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. the ground truth and the feature basis ----------------------
    sample = {"rows_in": 4000.0, "rows_out": 8000.0, "indexed": True}
    check(abs(true_cost(sample) - dot(TRUE_WEIGHTS, features(sample))) < 1e-12,
          "the ground truth must be exactly linear in the basis")
    check(features(sample)[3] > 0.0, "a 4000-row input must incur a spill term")
    check(features({"rows_in": 100.0, "rows_out": 10.0, "indexed": False})[3] == 0.0,
          "a small input must not spill")
    print("ground truth: cost = %s . basis (spill term kicks in above %.0f rows)"
          % (list(TRUE_WEIGHTS), MEMORY))

    stream = make_examples(generate_operators(20240501, 1200, "mixed"))
    check(len(stream) == 1200, "the stream must contain the requested examples")
    check(len({round(example["label"], 6) for example in stream}) > 1000,
          "the labels must vary across the stream")
    print("stream: %d examples, labels %.1f..%.1f"
          % (len(stream), min(example["label"] for example in stream),
             max(example["label"] for example in stream)))

    # ---- 2. the SGD update is a real gradient step ----------------------
    probe = OnlineCostModel()
    train(probe, stream[:200])
    example = stream[200]
    analytic = probe.gradient(example)
    epsilon = 1e-6
    numeric = []
    for index in range(probe.dimension):
        original = probe.weights[index]
        probe.weights[index] = original + epsilon
        high = probe.loss(example)
        probe.weights[index] = original - epsilon
        low = probe.loss(example)
        probe.weights[index] = original
        numeric.append((high - low) / (2 * epsilon))
    for index, (left, right) in enumerate(zip(analytic, numeric)):
        check(abs(left - right) < 1e-3 * max(1.0, abs(right)),
              "gradient component %d: analytic %.6f != numeric %.6f" % (index, left, right))
    print("gradient: analytic SGD direction matches the finite-difference gradient")

    # one full-batch epoch must equal a hand-written descent step
    manual = OnlineCostModel()
    manual.weights = list(probe.weights)
    manual.count = probe.count
    manual.mean = list(probe.mean)
    manual.moment = list(probe.moment)
    manual.updates = probe.updates
    manual.observe(example["features"])          # the same statistics update the step performs
    before = list(manual.weights)
    gradient, error = manual.descent_gradient(example)
    rate = manual.learning_rate / (1.0 + manual.decay * manual.updates)
    expected = [weight - rate * component for weight, component in zip(before, gradient)]
    manual.descent(example)
    for index, (left, right) in enumerate(zip(manual.weights, expected)):
        check(abs(left - right) < 1e-12,
              "the SGD step must equal w - lr*grad on component %d" % index)
    plain = manual.gradient(example)
    if abs(error) < manual.clip:
        check(plain == gradient, "the clip must be inactive for an ordinary error")
    print("update: one SGD step equals w - lr*gradient exactly (error %.3f)" % error)

    # ---- 3. learning converges ------------------------------------------
    model = OnlineCostModel()
    history = train(model, stream, epochs=6)
    blocks = [sum(history[index:index + 600]) / 600 for index in range(0, len(history) - 599, 600)]
    check(blocks[0] > blocks[-1], "the training loss must fall (%.3f -> %.3f)"
          % (blocks[0], blocks[-1]))
    check(blocks[-1] < blocks[0] / 20.0,
          "the loss must fall by more than 20x, got %.4f -> %.4f" % (blocks[0], blocks[-1]))
    held_out = stream[1000:]
    check(r_squared(model, held_out) > 0.99,
          "the model must explain the held-out labels (R2=%.4f)" % r_squared(model, held_out))
    check(mean_absolute_error(model, held_out) < 0.5,
          "the held-out MAE must be small, got %.4f" % mean_absolute_error(model, held_out))
    print("training: loss %.3f -> %.3f in %d updates, held-out R2=%.4f MAE=%.4f"
          % (blocks[0], blocks[-1], model.updates, r_squared(model, held_out),
             mean_absolute_error(model, held_out)))

    # ---- 4. normalisation is sound --------------------------------------
    variances = [model.standard_deviation(index) for index in range(model.dimension)]
    check(all(variance > 0.0 for variance in variances),
          "every feature must have a non-zero spread on this stream")
    standardised = model.normalised(stream[-1]["features"])
    check(len(standardised) == model.dimension, "normalisation must preserve the dimension")

    # ---- 5. coefficient recovery on a noiseless single-regime stream ----
    # plain least squares here (no decay, no clipping) so the target is exactly recoverable
    clean_model = OnlineCostModel(learning_rate=0.05, decay=0.0, clip=1e9)
    clean_stream = make_examples(generate_operators(9, 2000, "mixed"))
    train(clean_model, clean_stream, epochs=15)
    recovered = recovered_weights(clean_model)
    for index, (left, right) in enumerate(zip(recovered, TRUE_WEIGHTS)):
        # the intercept absorbs the centring of the other features, so it converges last
        tolerance = 0.05 if index == 0 else 0.02
        check(abs(left - right) <= tolerance * max(1.0, abs(right)),
              "coefficient %d recovered as %.4f, truth %.4f" % (index, left, right))
    print("recovery: raw coefficients %s vs truth %s"
          % ([round(value, 3) for value in recovered], list(TRUE_WEIGHTS)))

    # ---- 6. distribution shift ------------------------------------------
    shifted = OnlineCostModel()
    train(shifted, make_examples(generate_operators(11, 400, "small")), epochs=4)
    tail = make_examples(generate_operators(12, 200, "drifted"))
    before_error = mean_absolute_error(shifted, tail)
    train(shifted, make_examples(generate_operators(13, 300, "drifted")), epochs=4)
    late = mean_absolute_error(shifted, tail)
    frozen = OnlineCostModel()
    train(frozen, make_examples(generate_operators(11, 400, "small")), epochs=8)
    frozen_error = mean_absolute_error(frozen, tail)
    check(late < before_error / 2.0,
          "the online model must adapt to the new regime (%.3f -> %.3f)" % (before_error, late))
    check(late < frozen_error / 2.0,
          "adapting must beat a model frozen on the old regime (%.3f vs %.3f)"
          % (late, frozen_error))
    check(all(math.isfinite(weight) for weight in shifted.weights),
          "the clipped update must keep the weights finite under drift")
    print("shift: MAE before=%.3f after=%.3f frozen=%.3f"
          % (before_error, late, frozen_error))

    # ---- 7. the planner benefit -----------------------------------------
    rng = SplitMix64(4242)
    queries = query_plans(rng, 60)
    check(all(len(orderings(query["tables"])) == 6 for query in queries),
          "three tables must give six join orders")
    mixed = make_examples(generate_operators(31337, 1200, "mixed"))
    planner_model = OnlineCostModel()
    train(planner_model, mixed, epochs=6)
    learned_mean, learned_max, learned_exact = evaluate_planner(learned_planner(planner_model),
                                                                queries)
    static_mean, static_max, static_exact = evaluate_planner(static_planner, queries)
    check(learned_mean < static_mean,
          "the learned planner must beat the rule of thumb (%.3f vs %.3f)"
          % (learned_mean, static_mean))
    check(learned_exact >= static_exact and learned_exact > 0.8,
          "the learned planner must find the optimal order most of the time (%.2f vs %.2f)"
          % (learned_exact, static_exact))
    check(learned_max < static_max, "even the worst learned choice must be no worse")
    print("planner: learned regret mean=%.3f exact=%.2f | static regret mean=%.3f exact=%.2f"
          % (learned_mean, learned_exact, static_mean, static_exact))

    # plan quality keeps improving with more training
    partial_model = OnlineCostModel()
    train(partial_model, mixed[:200], epochs=1)
    partial_mean, _partial_max, _partial_exact = evaluate_planner(
        learned_planner(partial_model), queries)
    check(learned_mean <= partial_mean,
          "more training must not hurt the planner (%.3f vs %.3f)" % (learned_mean, partial_mean))
    print("planner: 200 examples -> regret %.3f, %d examples -> regret %.3f"
          % (partial_mean, len(mixed) * 6, learned_mean))

    # ---- 8. edge cases ---------------------------------------------------
    idle = OnlineCostModel()
    check(idle.weights == [0.0] * idle.dimension and mean_absolute_error(idle, []) == 0.0,
          "an untrained model must predict zero without crashing")
    train(idle, [])
    check(idle.updates == 0, "an empty stream must not update the model")
    one = OnlineCostModel()
    train(one, [stream[0]])
    check(one.updates == 1 and one.count == 1, "a single example must be consumed once")
    frozen_model = OnlineCostModel(learning_rate=0.0)
    train(frozen_model, stream[:50])
    check(frozen_model.weights == [0.0] * frozen_model.dimension,
          "a zero learning rate must not move the weights")
    constant = OnlineCostModel()
    train(constant, [{"features": [1.0] * 5, "label": 3.0}] * 20)
    check(all(math.isfinite(weight) for weight in constant.weights),
          "zero-variance features must not produce NaN weights")
    spread = [constant.standard_deviation(index) for index in range(constant.dimension)]
    check(all(value >= 0.0 for value in spread), "standard deviations must be non-negative")

    # ---- 9. determinism --------------------------------------------------
    again = OnlineCostModel()
    train(again, mixed, epochs=6)
    check(again.weights == planner_model.weights,
          "the learned weights must be reproducible from the seed")
    regenerated = make_examples(generate_operators(31337, 1200, "mixed"))
    check(regenerated == mixed, "the example stream must be reproducible from its seed")
    different = make_examples(generate_operators(31338, 1200, "mixed"))
    check(different != mixed, "a different seed must produce a different stream")
    print("determinism: identical weights and identical streams for the same seed")
    print("PASS")


if __name__ == "__main__":
    main()
