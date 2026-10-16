"""Task 866 - Implement feature attribution runtime (SHAP-like) scalable for tabular models.  (Tier: Master, Beyond - Machine Learning & AI)

A serving runtime answers "why did the model say that?" for tabular requests.  A model is a
quadratic polynomial in six features, the background distribution is an explicit list of rows,
and the value of a coalition is the exact conditional mean v(S) = E[f(x_S, X_S)] over those rows.
Because the model is quadratic and the background is small, coalitions can be enumerated and the
answer the runtime is supposed to approximate - the Shapley value of every feature - can be
computed independently by the textbook formula, so the runtime can be scored rather than trusted.

The runtime explains a request with KernelSHAP under an explicit coalition budget: the empty and
full coalitions are always evaluated, and the rest of the budget is spent on subsets drawn with
probability proportional to the Shapley kernel weights, after which the explanation is the
solution of the constrained weighted least squares problem (attributions must sum to
v(N) - v(empty)).  Everything is deterministic: the request stream, the background, the feature
draws and the coalition sampler all come from splitmix64 streams, so no thread, socket, clock or
unseeded randomness is involved, and a budget is a count of coalitions rather than a duration.

What is simplified: models are quadratic rather than boosted trees; the background is a small
fixed sample instead of a density estimate; latency is accounted as coalition evaluations (each
one batched over the background) instead of wall-clock time; and the least squares system is
solved directly instead of with an iterative solver.

Invariants verified (each against the exact Shapley oracle or a negative control):
  * efficiency - exact and sampled attributions sum to v(N) - v(empty) exactly, for every budget,
    because the solver is constrained;
  * exactness - KernelSHAP over all 2^6 coalitions reproduces the textbook Shapley values to
    within 1e-8, and the enumerated values satisfy the dummy, symmetry and linearity axioms;
  * consistency - raising one coefficient by a constant shifts exactly that feature's attribution
    by coefficient x (x - E[x]) and leaves every other attribution untouched, which is the exact
    invariant a monotonicity claim is built on;
  * estimator quality - the sampled coalition sizes follow the Shapley kernel distribution, and
    the explanation error against the oracle shrinks as the budget grows and ends inside
    tolerance, while the union of empty and full coalitions is always evaluated;
  * cost bound - the number of coalition evaluations never exceeds the budget plus the two
    mandatory coalitions, model evaluations are exactly background-size times that, repeated
    identical requests are served from cache without extra evaluations, and a budget that would
    overrun a hard cap raises instead of silently spending more;
  * determinism and validation - the same seed reproduces the same coalitions and attributions,
    and malformed requests (unknown feature, wrong mask length, empty background, negative
    budget) are rejected.
"""

import math

MASK64 = (1 << 64) - 1
FEATURES = 6
BACKGROUND_ROWS = 6
BUDGETS = (6, 12, 24, 48, 96, 192)
TOLERANCE = 0.05
RIDGE = 1e-12


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def uniform(self):
        return self.next_u64() / float(1 << 64)

    def below(self, bound):
        return self.next_u64() % bound


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting: deterministic, and enough for a 7x7 system."""
    size = len(rhs)
    grid = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: (abs(grid[row][column]), -row))
        if abs(grid[pivot][column]) < 1e-14:
            raise AssertionError("singular system")
        grid[column], grid[pivot] = grid[pivot], grid[column]
        for row in range(size):
            if row == column:
                continue
            factor = grid[row][column] / grid[column][column]
            if factor == 0.0:
                continue
            for entry in range(column, size + 1):
                grid[row][entry] -= factor * grid[column][entry]
    return [grid[index][size] / grid[index][index] for index in range(size)]


class QuadraticModel:
    """f(x) = base + sum a_j x_j + sum b_jk x_j x_k, plus an evaluation counter."""

    def __init__(self, name, base, linear, quadratic):
        self.name = name
        self.base = base
        self.linear = list(linear)
        self.quadratic = list(quadratic)
        self.evaluations = 0

    def evaluate(self, row):
        self.evaluations += 1
        total = self.base
        for index, value in enumerate(row):
            total += self.linear[index] * value
        for (left, right, coefficient) in self.quadratic:
            total += coefficient * row[left] * row[right]
        return total

    def mask(self, mask, request, background_row):
        return [request[index] if mask[index] else background_row[index]
                for index in range(len(self.linear))]

    def coalition_value(self, mask, request, background):
        total = 0.0
        for row in background:
            total += self.evaluate(self.mask(mask, request, row))
        return total / len(background)


def masks_of_size(size, width):
    result = []
    for bits in range(1 << width):
        chosen = [index for index in range(width) if bits >> index & 1]
        if len(chosen) == size:
            result.append(tuple(1 if index in chosen else 0 for index in range(width)))
    return result


def kernel_weights(size, width):
    """Shapley kernel weight for one coalition of this size."""
    return (width - 1) / (math.comb(width, size) * size * (width - size))


def size_distribution(width):
    """The kernel weights marginalised onto coalition size, normalised."""
    raw = [kernel_weights(size, width) * math.comb(width, size) for size in range(1, width)]
    total = sum(raw)
    return [value / total for value in raw]


class AttributionRuntime:
    """Explains a request under a coalition budget, and accounts for every evaluation."""

    def __init__(self, model, background, seed, hard_cap=None):
        if not background:
            raise ValueError("the background set must not be empty")
        width = len(model.linear)
        if any(len(row) != width for row in background):
            raise ValueError("background rows must have %d features" % width)
        self.model = model
        self.background = [list(row) for row in background]
        self.width = width
        self.seed = seed
        self.hard_cap = hard_cap
        self.coalition_evaluations = 0
        self.cache = {}
        self.attributions = {}
        self.start_evaluations = model.evaluations

    # -- coalition values ----------------------------------------------------
    def value(self, mask, request):
        if len(mask) != self.width:
            raise ValueError("a coalition mask must describe %d features" % self.width)
        key = (tuple(mask), tuple(request))
        if key in self.cache:
            return self.cache[key]
        if self.hard_cap is not None and self.coalition_evaluations >= self.hard_cap:
            raise RuntimeError("coalition budget exhausted (%d)" % self.hard_cap)
        self.coalition_evaluations += 1
        result = self.model.coalition_value(mask, request, self.background)
        self.cache[key] = result
        return result

    def model_evaluations(self):
        return self.model.evaluations - self.start_evaluations

    # -- the oracle: textbook Shapley values over all subsets ----------------
    def exact_shapley(self, request):
        width = self.width
        factorial = math.factorial
        values = {}
        for size in range(width + 1):
            for mask in masks_of_size(size, width):
                values[mask] = self.value(mask, request)
        result = []
        for feature in range(width):
            total = 0.0
            others = [index for index in range(width) if index != feature]
            for bits in range(1 << len(others)):
                subset = [others[position] for position in range(len(others))
                          if bits >> position & 1]
                size = len(subset)
                with_feature = tuple(1 if index in subset or index == feature else 0
                                     for index in range(width))
                without = tuple(1 if index in subset else 0 for index in range(width))
                weight = factorial(size) * factorial(width - size - 1) / factorial(width)
                total += weight * (values[with_feature] - values[without])
            result.append(total)
        return result

    # -- the runtime: KernelSHAP with a coalition budget ---------------------
    def sample_coalitions(self, budget, strategy="kernel"):
        width = self.width
        rng = SplitMix64(fnv1a(self.seed, strategy, budget, self.coalition_evaluations,
                               len(self.cache)))
        distribution = size_distribution(width)
        coalitions = [(0,) * width, (1,) * width]
        for _draw in range(budget):
            if strategy == "kernel":
                pick = rng.uniform()
                size = width - 1
                cumulative = 0.0
                for index, weight in enumerate(distribution):
                    cumulative += weight
                    if pick < cumulative:
                        size = index + 1
                        break
            elif strategy == "uniform":
                size = 1 + rng.below(width - 1)
            else:
                raise ValueError("unknown sampling strategy %s" % strategy)
            options = masks_of_size(size, width)
            coalitions.append(options[rng.below(len(options))])
        return coalitions

    def explain(self, request, budget, strategy="kernel"):
        """The KernelSHAP solution of the constrained weighted least squares problem."""
        if budget < 0:
            raise ValueError("the coalition budget must not be negative")
        if len(request) != self.width:
            raise ValueError("a request must have %d features" % self.width)
        coalitions = self.sample_coalitions(budget, strategy)
        values = {mask: self.value(mask, request) for mask in coalitions}
        empty = self.value((0,) * self.width, request)
        full = self.value((1,) * self.width, request)
        share = (full - empty) / self.width
        width = self.width
        matrix = [[RIDGE if row == column else 0.0 for column in range(width + 1)]
                  for row in range(width + 1)]
        rhs = [0.0] * (width + 1)
        for mask in sorted(values):
            size = sum(mask)
            if size in (0, width):
                continue
            weight = kernel_weights(size, width)
            target = values[mask] - empty - size * share
            for left in range(width):
                if not mask[left]:
                    continue
                rhs[left] += weight * target
                for right in range(width):
                    if mask[right]:
                        matrix[left][right] += weight
        for index in range(width):
            matrix[index][width] = 1.0
            matrix[width][index] = 1.0
        matrix[width][width] = 0.0
        rhs[width] = 0.0
        solution = solve(matrix, rhs)
        explanation = [solution[index] + share for index in range(width)]
        self.attributions[(tuple(request), budget, strategy)] = explanation
        return explanation

    def explain_cached(self, request, budget, strategy="kernel"):
        key = (tuple(request), budget, strategy)
        if key in self.attributions:
            return self.attributions[key]
        return self.explain(request, budget, strategy)


def fnv1a(*parts):
    digest = 0xCBF29CE484222325
    for part in parts:
        for character in str(part):
            digest = ((digest ^ ord(character)) * 0x100000001B3) & MASK64
    return digest


def build_model(seed, zero_feature=None):
    rng = SplitMix64(seed)
    linear = [round(2.0 * rng.uniform() - 1.0, 6) for _ in range(FEATURES)]
    quadratic = []
    for left in range(FEATURES):
        for right in range(left, FEATURES):
            if rng.uniform() < 0.5:
                quadratic.append((left, right, round(1.0 * rng.uniform() - 0.5, 6)))
    if zero_feature is not None:
        linear[zero_feature] = 0.0
        quadratic = [term for term in quadratic
                     if zero_feature not in (term[0], term[1])]
    return QuadraticModel("f%s" % seed, 0.5, linear, quadratic)


def build_background(seed, rows=BACKGROUND_ROWS):
    rng = SplitMix64(seed)
    return [[round(0.5 + rng.uniform(), 6) for _ in range(FEATURES)] for _ in range(rows)]


def build_request(seed):
    rng = SplitMix64(seed)
    return [round(0.25 + rng.uniform(), 6) for _ in range(FEATURES)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def max_error(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def main():
    model = build_model(11)
    background = build_background(12)
    request = build_request(13)
    check(FEATURES == 6 and len(background) == BACKGROUND_ROWS,
          "the fixture dimensions must be the ones the runtime is sized for")
    check(any(term[0] != term[1] for term in model.quadratic),
          "the fixture model must contain interactions, otherwise attribution is trivial")

    # ---- 1. the exact oracle and its axioms --------------------------------
    oracle = AttributionRuntime(model, background, seed=101)
    exact = oracle.exact_shapley(request)
    empty = oracle.value((0,) * FEATURES, request)
    full = oracle.value((1,) * FEATURES, request)
    check(len(oracle.cache) == 1 << FEATURES,
          "the oracle must evaluate every one of the 2^%d coalitions" % FEATURES)
    check(abs(sum(exact) - (full - empty)) < 1e-9,
          "efficiency: the attributions must sum to v(full) - v(empty) (%.9f vs %.9f)"
          % (sum(exact), full - empty))
    dummy_model = build_model(11, zero_feature=4)
    dummy_runtime = AttributionRuntime(dummy_model, background, seed=101)
    dummy_exact = dummy_runtime.exact_shapley(request)
    check(dummy_exact[4] == 0.0,
          "dummy: a feature that never enters the model must get exactly zero (%.12f)"
          % dummy_exact[4])
    check(abs(sum(dummy_exact) - (dummy_runtime.value((1,) * FEATURES, request)
                                  - dummy_runtime.value((0,) * FEATURES, request))) < 1e-9,
          "the dummy model must still satisfy efficiency")
    check(max_error(dummy_exact, exact) > 0.01,
          "removing the feature must really change the other attributions")
    symmetric_background = [[row[0], row[1], row[1], row[3], row[4], row[5]]
                            for row in background]
    symmetric_request = [request[0], request[1], request[1], request[3], request[4], request[5]]
    symmetric_linear = list(model.linear)
    symmetric_linear[2] = symmetric_linear[1]
    swap = {1: 2, 2: 1}
    symmetric_quadratic = list(model.quadratic)
    for (left, right, coefficient) in model.quadratic:
        if 1 in (left, right) or 2 in (left, right):
            symmetric_quadratic.append((swap.get(left, left), swap.get(right, right),
                                        coefficient))
    symmetric = QuadraticModel("symmetric", model.base, symmetric_linear, symmetric_quadratic)
    symmetric_runtime = AttributionRuntime(symmetric, symmetric_background, seed=101)
    symmetric_exact = symmetric_runtime.exact_shapley(symmetric_request)
    check(abs(symmetric_exact[1] - symmetric_exact[2]) < 1e-9,
          "symmetry: features 1 and 2 play the same role and must be attributed equally "
          "(%.9f vs %.9f)" % (symmetric_exact[1], symmetric_exact[2]))
    mirrored = [list(row) for row in symmetric_background]
    for row in mirrored:
        row[1], row[2] = row[2], row[1]
    for row, twin in zip(symmetric_background, mirrored):
        check(abs(symmetric.evaluate(row) - symmetric.evaluate(twin)) < 1e-12,
              "the fixture model must really be invariant under swapping features 1 and 2")
    second = build_model(99)
    combined = QuadraticModel("f+g", model.base + second.base,
                              [a + b for a, b in zip(model.linear, second.linear)],
                              [(l, r, c) for (l, r, c) in model.quadratic]
                              + [(l, r, c) for (l, r, c) in second.quadratic])
    combined_exact = AttributionRuntime(combined, background, seed=101).exact_shapley(request)
    second_exact = AttributionRuntime(second, background, seed=101).exact_shapley(request)
    check(max_error(combined_exact, [a + b for a, b in zip(exact, second_exact)]) < 1e-9,
          "linearity: attributions of f + g must be the sum of the attributions")
    check(all(math.isfinite(value) for value in combined_exact),
          "every attribution of the combined model must be a finite number")
    print("oracle: efficiency %.9f, dummy %.12f, symmetry %.9f vs %.9f, linearity exact"
          % (full - empty, dummy_exact[4], symmetric_exact[1], symmetric_exact[2]))

    # ---- 2. consistency: a stronger feature is attributed more -------------
    feature = 2
    weak = build_model(11)
    strong = QuadraticModel("strong", weak.base, list(weak.linear), list(weak.quadratic))
    strong.linear[feature] = weak.linear[feature] + 1.5
    weak_values = AttributionRuntime(weak, background, seed=101).exact_shapley(request)
    strong_values = AttributionRuntime(strong, background, seed=101).exact_shapley(request)
    mean_background = sum(row[feature] for row in background) / len(background)
    predicted = 1.5 * (request[feature] - mean_background)
    check(abs((strong_values[feature] - weak_values[feature]) - predicted) < 1e-9,
          "the attribution of a purely linear term must move by coefficient x (x - E[x]) "
          "(%.9f vs %.9f)" % (strong_values[feature] - weak_values[feature], predicted))
    check(abs(predicted) > 1e-6,
          "the consistency fixture must actually move the attribution (%.6f)" % predicted)
    check(max_error([value for index, value in enumerate(strong_values) if index != feature],
                    [value for index, value in enumerate(weak_values) if index != feature]) < 1e-9,
          "only the changed feature's attribution may move")
    print("consistency: feature %d attribution %.4f -> %.4f (predicted shift %.4f)"
          % (feature, weak_values[feature], strong_values[feature], predicted))

    # ---- 3. exact KernelSHAP over all coalitions ---------------------------
    everything = 1 << FEATURES
    all_coalitions = [tuple(1 if bits >> index & 1 else 0 for index in range(FEATURES))
                      for bits in range(everything)]
    full_runtime = AttributionRuntime(model, background, seed=101)
    full_runtime.sample_coalitions = lambda budget, strategy="kernel": all_coalitions
    solved = full_runtime.explain(request, everything)
    check(max_error(solved, exact) < 1e-8,
          "KernelSHAP over every coalition must reproduce the oracle (max error %.2e)"
          % max_error(solved, exact))
    check(abs(sum(solved) - (full - empty)) < 1e-8,
          "the constrained solve must satisfy efficiency as well")
    print("exhaustive KernelSHAP: max error vs the oracle %.2e over %d coalitions"
          % (max_error(solved, exact), everything))

    # ---- 4. the sampler follows the Shapley kernel -------------------------
    sampler = AttributionRuntime(model, background, seed=2024)
    drawn = sampler.sample_coalitions(20000)
    check(len(drawn) == 20002 and drawn[0] == (0,) * FEATURES and drawn[1] == (1,) * FEATURES,
          "the empty and full coalitions must always be evaluated")
    sizes = [0] * (FEATURES + 1)
    for mask in drawn:
        sizes[sum(mask)] += 1
    interior = sum(sizes[1:FEATURES])
    observed = [sizes[size] / interior for size in range(1, FEATURES)]
    promised = size_distribution(FEATURES)
    check(max(abs(a - b) for a, b in zip(observed, promised)) < 0.03,
          "the sampled sizes must follow the kernel distribution (%s vs %s)"
          % ([round(value, 3) for value in observed], [round(value, 3) for value in promised]))
    check(observed[0] > observed[2] and observed[-1] > observed[2],
          "the kernel must concentrate on small and large coalitions: %s"
          % [round(value, 3) for value in observed])
    print("sampler: kernel size distribution %s vs observed %s"
          % ([round(value, 3) for value in promised], [round(value, 3) for value in observed]))

    # ---- 5. budget, error and cost accounting ------------------------------
    queries = [build_request(1000 + index) for index in range(8)]
    curve = {}
    for budget in BUDGETS:
        total = 0.0
        for index, query in enumerate(queries):
            runtime = AttributionRuntime(model, background, seed=500 + index)
            explanation = runtime.explain(query, budget)
            if budget == BUDGETS[-1]:
                check(runtime.coalition_evaluations <= budget + 2,
                      "the runtime must stay inside its budget (%d evaluations for budget %d)"
                      % (runtime.coalition_evaluations, budget))
                check(runtime.model_evaluations()
                      == runtime.coalition_evaluations * len(background),
                      "each coalition is one batched model call per background row (%d vs %d)"
                      % (runtime.model_evaluations(),
                         runtime.coalition_evaluations * len(background)))
                empty_v = runtime.value((0,) * FEATURES, query)
                full_v = runtime.value((1,) * FEATURES, query)
                check(abs(sum(explanation) - (full_v - empty_v)) < 1e-8,
                      "sampled explanations must satisfy efficiency exactly, whatever the budget")
            reference = AttributionRuntime(model, background, seed=101).exact_shapley(query)
            total += max_error(explanation, reference)
        curve[budget] = total / len(queries)
    check(all(curve[a] >= curve[b] - 1e-9 for a, b in zip(BUDGETS, BUDGETS[1:])),
          "the mean error must shrink with the budget: %s" % curve)
    check(curve[BUDGETS[-1]] < curve[BUDGETS[0]],
          "the largest budget must beat the smallest (%.4f vs %.4f)"
          % (curve[BUDGETS[-1]], curve[BUDGETS[0]]))
    check(curve[BUDGETS[-1]] <= TOLERANCE,
          "the budget we ship must be inside tolerance (%.4f)" % curve[BUDGETS[-1]])
    exact_runtime = AttributionRuntime(model, background, seed=101)
    exact_runtime.exact_shapley(queries[0])
    started = exact_runtime.coalition_evaluations
    exact_runtime.exact_shapley(queries[0])
    check(exact_runtime.coalition_evaluations == started,
          "a repeated explanation must be served from cache without new evaluations")
    print("budget curve: %s (mean max-error over %d requests, tolerance %.2f)"
          % ({key: round(value, 4) for key, value in curve.items()}, len(queries), TOLERANCE))

    # ---- 6. the estimator choice is measured -------------------------------
    kernel_errors = []
    uniform_errors = []
    for index, query in enumerate(queries):
        reference = AttributionRuntime(model, background, seed=101).exact_shapley(query)
        kernel_errors.append(max_error(AttributionRuntime(model, background,
                                                          seed=700 + index).explain(query, 24),
                                       reference))
        uniform_errors.append(max_error(AttributionRuntime(model, background,
                                                           seed=700 + index)
                                        .explain(query, 24, strategy="uniform"), reference))
    kernel_mean = sum(kernel_errors) / len(kernel_errors)
    uniform_mean = sum(uniform_errors) / len(uniform_errors)
    check(kernel_mean < uniform_mean,
          "kernel-weighted sampling must beat uniform sampling at the same budget (%.4f vs %.4f)"
          % (kernel_mean, uniform_mean))
    print("sampling: kernel weights %.4f mean error vs uniform sizes %.4f at budget 24"
          % (kernel_mean, uniform_mean))

    # ---- 7. edge cases and validation --------------------------------------
    single = QuadraticModel("single", 1.0, [3.0], [])
    single_background = [[1.0], [2.0]]
    single_runtime = AttributionRuntime(single, single_background, seed=3)
    single_explanation = single_runtime.explain([4.0], 0)
    check(abs(single_explanation[0] - (single.evaluate([4.0]) - single.evaluate([1.5]))) < 1e-9,
          "with one feature the attribution is the whole difference")
    flat = QuadraticModel("flat", 2.5, [0.0] * FEATURES, [])
    flat_runtime = AttributionRuntime(flat, background, seed=4)
    flat_explanation = flat_runtime.explain_cached(request, 32)
    check(all(value == 0.0 for value in flat_explanation),
          "a constant model attributes nothing: %s" % flat_explanation)
    capped = AttributionRuntime(model, background, seed=5, hard_cap=4)
    overrun = False
    try:
        capped.explain(request, 8)
    except RuntimeError:
        overrun = True
    check(overrun, "exceeding the hard cap must raise instead of overspending")
    guard = AttributionRuntime(model, background, seed=6)
    check(guard.explain_cached(request, 32) == guard.explain_cached(request, 32),
          "a cached explanation must be returned unchanged")
    failures = 0
    for action in (lambda: AttributionRuntime(model, [], 1),
                   lambda: AttributionRuntime(model, [[0.0] * 5], 1),
                   lambda: AttributionRuntime(model, background, 1).explain(request, -1),
                   lambda: AttributionRuntime(model, background, 1).explain(request[:3], 8),
                   lambda: AttributionRuntime(model, background, 1).value((1, 1), request),
                   lambda: AttributionRuntime(model, background, 1).explain(request, 8,
                                                                           strategy="magic")):
        try:
            action()
        except (ValueError, RuntimeError):
            failures += 1
    check(failures == 6, "every malformed request must be rejected (%d of 6)" % failures)
    print("edge cases: one feature, constant model, hard cap, cache, %d rejected requests"
          % failures)

    # ---- 8. determinism ----------------------------------------------------
    left = AttributionRuntime(model, background, seed=31415)
    right = AttributionRuntime(model, background, seed=31415)
    check(left.sample_coalitions(64) == right.sample_coalitions(64),
          "the same seed must draw the same coalitions")
    check(left.explain(request, 64) == right.explain(request, 64),
          "the same seed must produce the same explanation")
    other = AttributionRuntime(model, background, seed=27182)
    check(other.sample_coalitions(64) != left.sample_coalitions(64),
          "a different seed must draw different coalitions")
    check(max_error(other.explain(request, 64), exact) <= TOLERANCE,
          "a different seed must also stay inside tolerance at budget 64")
    check(max_error(other.explain(request, 64), left.explain(request, 64)) > 0.0,
          "independent seeds must produce genuinely different estimates")
    print("determinism: identical coalitions and explanations for seed 31415, a different seed "
          "still inside tolerance")
    print("PASS")


if __name__ == "__main__":
    main()
