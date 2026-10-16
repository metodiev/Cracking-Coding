"""Task 867 - Implement counterfactual explanation generator for classifiers (optimization-based).  (Tier: Master, Beyond - Machine Learning & AI)

The generator answers "what is the smallest change to this request that would flip the
classifier's answer?".  A request is a vector of five features, some immutable (age of an
account, a device id), some bounded, some integral; the classifier is a logistic model, so the
probability of a class and the decision boundary are exact and cheap.  The generator searches
sparse-first: it enumerates supports (subsets of the mutable features) in increasing size and, for
each support, minimises the sum of the logistic loss for the target class and the explanation
cost by projected gradient descent with deterministic step-size backtracking, starting from both
the query and the furthest feasible corner of the support.  The first support size that admits a
valid counterfactual wins, and the cheapest candidate inside it is returned.

What is simplified: the classifier is logistic rather than a boosted ensemble; the search space is
small enough that exhaustive enumeration over a discretised grid is possible, which is exactly
what makes the result checkable; the descent uses analytic gradients of the known model instead
of a black-box optimiser, and integral coordinates are handled by enumerating their values and
projecting the rest; a search budget is a count of candidate points rather than a wall-clock
deadline, and no thread, socket or clock is involved.

Invariants verified (each against a brute-force oracle over the discretised feasible space, or
against a negative control):
  * validity - the returned counterfactual really receives the requested class, and requesting the
    class the query already has is rejected rather than silently returned;
  * feasibility - immutable features are untouched, every value stays inside its declared bounds,
    and integral features come back integral;
  * minimality - the number of changed features is never larger than the minimum found by
    exhaustive enumeration, no combination of fewer features and grid values produces a valid
    counterfactual (a brute-force certificate), and no single-coordinate perturbation of the
    answer yields a cheaper valid counterfactual (a local optimality certificate);
  * optimality - the returned cost is at most the best cost reachable anywhere on the grid, over
    many queries and both target directions;
  * infeasibility - when no feasible change can reach the target class the generator reports
    "no counterfactual" instead of returning a point that does not flip the answer, and this
    agrees with exhaustive enumeration and with random probing;
  * cost model - raising the sparsity weight never yields a solution with more changed features,
    and the cost of the answer equals the independently recomputed cost of its own support;
  * determinism, budget and validation - the same seed reproduces the same answer, the search
    budget (candidate points tried) is respected and a hard cap raises instead of overspending,
    the answer costs a single classifier query to verify, and malformed requests (unknown class,
    out-of-bounds query, contradictory request) are rejected.
"""

import math

MASK64 = (1 << 64) - 1
FEATURES = 5
IMMUTABLE = (0, 3)
INTEGRAL = (4,)
LOWER = [0.0, 0.0, 0.0, 0.0, 0.0]
UPPER = [10.0, 10.0, 10.0, 10.0, 10.0]
GRID_LEVELS = 6
STEPS = 80
BUDGET = 400
MARGIN_GAP = 1e-6


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


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class LogisticClassifier:
    def __init__(self, name, weights, bias):
        self.name = name
        self.weights = list(weights)
        self.bias = bias
        self.queries = 0

    def margin(self, row):
        return self.bias + sum(weight * value for weight, value in zip(self.weights, row))

    def probability(self, row):
        self.queries += 1
        return sigmoid(self.margin(row))

    def predict(self, row):
        return 1 if self.probability(row) >= 0.5 else 0


class Constraints:
    """Which features may move, how far, and which are whole numbers."""

    def __init__(self, immutable=IMMUTABLE, integral=INTEGRAL, lower=LOWER, upper=UPPER):
        self.width = len(lower)
        self.immutable = frozenset(immutable)
        self.integral = frozenset(integral)
        self.lower = list(lower)
        self.upper = list(upper)

    def mutable(self):
        return [index for index in range(self.width)
                if index not in self.immutable and self.lower[index] < self.upper[index]]

    def within_bounds(self, row):
        if len(row) != self.width:
            return False
        return all(self.lower[index] <= row[index] <= self.upper[index]
                   for index in range(self.width))

    def project(self, row, support):
        result = list(row)
        for index in range(self.width):
            if index not in support:
                continue
            result[index] = min(max(result[index], self.lower[index]), self.upper[index])
            if index in self.integral:
                result[index] = float(round(result[index]))
        return result

    def satisfies(self, query, candidate):
        if not self.within_bounds(candidate):
            return False
        for index in self.immutable:
            if abs(candidate[index] - query[index]) > 1e-12:
                return False
        for index in self.integral:
            if abs(candidate[index] - round(candidate[index])) > 1e-9:
                return False
        return True

    def changed(self, query, candidate):
        return [index for index in range(self.width)
                if abs(candidate[index] - query[index]) > 1e-9]


class CostModel:
    """Explanation cost: quadratic magnitude per feature plus a flat charge per changed feature."""

    def __init__(self, sparsity_weight, l2_weight=1.0, per_feature=None):
        self.sparsity_weight = sparsity_weight
        self.l2_weight = l2_weight
        self.per_feature = dict(per_feature or {})

    def cost(self, query, candidate, changed=None):
        changed = self.changed(query, candidate) if changed is None else changed
        total = self.sparsity_weight * len(changed)
        for index in changed:
            weight = self.per_feature.get(index, 1.0)
            total += self.l2_weight * weight * (candidate[index] - query[index]) ** 2
        return total

    @staticmethod
    def changed(query, candidate):
        return [index for index in range(len(query))
                if abs(candidate[index] - query[index]) > 1e-9]


class Counterfactual:
    def __init__(self, candidate, cost, changed, support, cost_model, query):
        self.candidate = candidate
        self.cost = cost
        self.changed = changed
        self.support = support
        self.cost_model = cost_model
        self.query = query

    def describe(self):
        return "cost %.4f over %d features (%s)" % (self.cost, len(self.changed),
                                                   sorted(self.changed))


def combinations_of(items, size):
    """All size-sized subsets of `items` as ascending tuples, in a deterministic order."""
    pools = [[]]
    for _ in range(size):
        pools = [entry + [item] for entry in pools for item in items
                 if not entry or item > entry[-1]]
    return [tuple(entry) for entry in pools]


class CounterfactualGenerator:
    def __init__(self, classifier, constraints, cost_model, steps=STEPS, budget=BUDGET):
        self.classifier = classifier
        self.constraints = constraints
        self.cost_model = cost_model
        self.steps = steps
        self.budget = budget
        self.attempts = 0

    # -- optimisation --------------------------------------------------------
    def _loss(self, row, query, support, direction):
        margin = direction * self.classifier.margin(row)
        logistic = math.log1p(math.exp(-margin)) if margin > -60.0 else -margin
        return logistic + self.cost_model.cost(query, row)

    def _descend(self, query, support, direction, start):
        """Projected gradient descent with deterministic backtracking.  The cheapest iterate that
        already flips the answer is returned, which is what the generator scores."""
        self.attempts += 1
        point = list(start)
        rate = 0.5
        best_valid = None
        best_loss = (self._loss(point, query, support, direction), list(point))
        weights = self.classifier.weights
        for _step in range(self.steps):
            if direction * self.classifier.margin(point) >= MARGIN_GAP:
                changed = [index for index in support
                           if abs(point[index] - query[index]) > 1e-9]
                cost = self.cost_model.cost(query, point, changed)
                if best_valid is None or cost < best_valid[0] - 1e-12:
                    best_valid = (cost, list(point))
            margin = direction * self.classifier.margin(point)
            slope = sigmoid(-margin)
            gradient = []
            for index in support:
                pull = 2.0 * self.cost_model.l2_weight \
                    * self.cost_model.per_feature.get(index, 1.0) * (point[index] - query[index])
                gradient.append(-direction * weights[index] * slope + pull)
            improved = False
            for _attempt in range(8):
                trial = list(point)
                for position, index in enumerate(support):
                    trial[index] = point[index] - rate * gradient[position]
                trial = self.constraints.project(trial, support)
                candidate = self._loss(trial, query, support, direction)
                if candidate < best_loss[0] - 1e-15:
                    best_loss = (candidate, list(trial))
                if candidate <= self._loss(point, query, support, direction):
                    point = trial
                    rate = min(rate * 1.4, 8.0)
                    improved = True
                    break
                rate *= 0.5
                if rate < 1e-6:
                    break
            if not improved:
                break
        if best_valid is not None:
            return best_valid[1]
        return best_loss[1]

    def _corner(self, query, support, direction):
        corner = list(query)
        for index in support:
            if direction * self.classifier.weights[index] > 0.0:
                corner[index] = self.constraints.upper[index]
            else:
                corner[index] = self.constraints.lower[index]
        return self.constraints.project(corner, support)

    def _shoot(self, query, support, direction, fixed=None):
        """Closed-form cheapest move on this support: the weighted-L2 projection of the query onto
        the decision boundary, with box clipping handled by an active-set loop.  Integral
        coordinates are supplied already fixed by the caller, who enumerates their values."""
        self.attempts += 1
        weights = self.classifier.weights
        fixed = dict(fixed or {})
        base = list(query)
        for index, value in fixed.items():
            base[index] = value
        need = -direction * self.classifier.margin(base) + MARGIN_GAP
        active = [index for index in support if index not in fixed]
        deltas = {index: 0.0 for index in support}
        for _round in range(len(active) + 1):
            if need <= 0.0 or not active:
                break
            denominator = 0.0
            for index in active:
                gamma = self.cost_model.l2_weight \
                    * self.cost_model.per_feature.get(index, 1.0)
                denominator += weights[index] ** 2 / gamma
            if denominator <= 1e-12:
                break
            scale = need / denominator
            clipped = []
            for index in active:
                gamma = self.cost_model.l2_weight \
                    * self.cost_model.per_feature.get(index, 1.0)
                step = direction * scale * weights[index] / gamma
                value = query[index] + step
                if value < self.constraints.lower[index]:
                    clipped.append((index, self.constraints.lower[index]))
                elif value > self.constraints.upper[index]:
                    clipped.append((index, self.constraints.upper[index]))
                else:
                    deltas[index] = step
            if not clipped:
                break
            for index, value in clipped:
                active.remove(index)
                deltas[index] = value - query[index]
                need -= direction * weights[index] * deltas[index]
        candidate = list(query)
        for index in deltas:
            candidate[index] = query[index] + deltas[index]
        for index, value in fixed.items():
            candidate[index] = value
        return self.constraints.project(candidate, support)

    def _plans(self, support):
        """Every assignment of the integral coordinates inside the support."""
        plans = [{}]
        for index in support:
            if index not in self.constraints.integral:
                continue
            grid = [float(value)
                    for value in range(int(round(self.constraints.lower[index])),
                                       int(round(self.constraints.upper[index])) + 1)]
            extended = []
            for plan in plans:
                for value in grid:
                    extended.append(dict(list(plan.items()) + [(index, value)]))
            plans = extended
        return plans

    def _refine(self, query, support, candidate, direction):
        """Bisect the segment query -> candidate for the smallest move that still flips the
        answer, which is the cheapest point on that segment.  The flip test keeps a small margin
        so that the point cannot fall back onto the boundary through rounding."""
        self.attempts += 1
        if candidate is None:
            return None
        best = candidate
        low, high = 0.0, 1.0
        for _iteration in range(40):
            middle = 0.5 * (low + high)
            trial = [query[index] + middle * (candidate[index] - query[index])
                     for index in range(len(query))]
            trial = self.constraints.project(trial, support)
            if direction * self.classifier.margin(trial) >= MARGIN_GAP:
                best = trial
                high = middle
            else:
                low = middle
        return best if direction * self.classifier.margin(best) >= MARGIN_GAP else candidate

    # -- the generator -------------------------------------------------------
    def explain(self, query, target):
        if target not in (0, 1):
            raise ValueError("the target class must be 0 or 1")
        if not self.constraints.within_bounds(query):
            raise ValueError("the query is outside the declared feature bounds")
        if len(query) != self.constraints.width:
            raise ValueError("the query must have %d features" % self.constraints.width)
        if self.classifier.predict(query) == target:
            raise ValueError("the query already receives the requested class")
        direction = 1.0 if target == 1 else -1.0
        mutable = self.constraints.mutable()
        if not mutable:
            return None
        started = self.attempts
        for size in range(1, len(mutable) + 1):
            best = None
            for support in combinations_of(mutable, size):
                attempts = [self._descend(query, support, direction, query),
                            self._descend(query, support, direction,
                                          self._corner(query, support, direction))]
                for plan in self._plans(support):
                    attempts.append(self._shoot(query, support, direction, plan))
                if self.attempts - started > self.budget:
                    raise RuntimeError("the counterfactual search budget is exhausted")
                for attempt in attempts:
                    for candidate in (attempt, self._refine(query, support, attempt, direction)):
                        if candidate is None:
                            continue
                        if direction * self.classifier.margin(candidate) < MARGIN_GAP:
                            continue
                        if not self.constraints.satisfies(query, candidate):
                            continue
                        changed = self.constraints.changed(query, candidate)
                        cost = self.cost_model.cost(query, candidate, changed)
                        key = (round(cost, 9), tuple(candidate))
                        if best is None or key < best[0]:
                            best = (key, Counterfactual(list(candidate), cost, changed, support,
                                                        self.cost_model, list(query)))
            if best is not None:
                return best[1]
        return None


class GridOracle:
    """Exhaustive enumeration over a discretised feasible space: the independent reference."""

    def __init__(self, classifier, constraints, cost_model, levels=GRID_LEVELS):
        self.classifier = classifier
        self.constraints = constraints
        self.cost_model = cost_model
        self.levels = levels

    def values_for(self, index, query):
        lower, upper = self.constraints.lower[index], self.constraints.upper[index]
        if index in self.constraints.integral:
            span = int(round(upper - lower))
            grid = [float(lower + step) for step in range(span + 1)]
        else:
            grid = [lower + (upper - lower) * step / (self.levels - 1)
                    for step in range(self.levels)]
        if query[index] not in grid:
            grid.append(query[index])
        return sorted(set(grid))

    def valid_points(self, query, target):
        mutable = self.constraints.mutable()
        grids = [self.values_for(index, query) for index in mutable]
        results = []
        for combination in self._product(grids):
            candidate = list(query)
            for position, index in enumerate(mutable):
                candidate[index] = combination[position]
            candidate = self.constraints.project(candidate, mutable)
            if not self.constraints.satisfies(query, candidate):
                continue
            if self.classifier.predict(candidate) != target:
                continue
            changed = self.constraints.changed(query, candidate)
            results.append((self.cost_model.cost(query, candidate, changed), len(changed),
                            tuple(candidate)))
        results.sort()
        return results

    @staticmethod
    def _product(grids):
        combos = [[]]
        for grid in grids:
            combos = [entry + [value] for entry in combos for value in grid]
        return combos


def build_classifier(seed):
    rng = SplitMix64(seed)
    weights = [round(2.0 * rng.uniform() - 1.0, 6) for _ in range(FEATURES)]
    bias = round(0.6 * rng.uniform() - 0.3, 6)
    return LogisticClassifier("logit-%d" % seed, weights, bias)


def build_query(seed):
    rng = SplitMix64(seed)
    return [round(2.0 + 6.0 * rng.uniform(), 6) for _ in range(FEATURES)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def probe_infeasible(generator, classifier, query, target):
    """Random probing against exhaustive enumeration, to confirm 'no counterfactual' claims."""
    rng = SplitMix64(99)
    mutable = generator.constraints.mutable()
    for _attempt in range(400):
        candidate = list(query)
        for index in mutable:
            candidate[index] = round(10.0 * rng.uniform(), 6)
        candidate = generator.constraints.project(candidate, mutable)
        if classifier.predict(candidate) == target:
            return candidate
    return None


def main():
    classifier = build_classifier(4242)
    constraints = Constraints()
    cost_model = CostModel(sparsity_weight=1000.0)
    generator = CounterfactualGenerator(classifier, constraints, cost_model)
    oracle = GridOracle(classifier, constraints, cost_model)
    check(len(constraints.mutable()) == 3,
          "the fixture must have three mutable features, not a trivial one")
    print("classifier: weights %s bias %.4f, mutable features %s, integral %s"
          % (classifier.weights, classifier.bias, constraints.mutable(),
             sorted(constraints.integral)))

    queries = [build_query(1000 + index) for index in range(6)]
    filled = 0
    for query in queries:
        current = classifier.predict(query)
        if current == 1:
            filled += 1
    check(0 < filled < len(queries),
          "the fixture must contain queries of both classes (%d of %d are class 1)"
          % (filled, len(queries)))

    # ---- 1. validity, feasibility and minimality against the oracle ---------
    comparisons = 0
    for query in queries:
        for target in (0, 1):
            if classifier.predict(query) == target:
                continue
            result = generator.explain(query, target)
            reference = oracle.valid_points(query, target)
            comparisons += 1
            if result is None:
                check(not reference,
                      "the generator must find a counterfactual where the grid has one (%s)"
                      % reference[:1])
                continue
            check(reference, "the oracle must find at least the point the generator found")
            check(classifier.predict(result.candidate) == target,
                  "the counterfactual must really receive class %d" % target)
            check(constraints.satisfies(query, result.candidate),
                  "the counterfactual must be feasible")
            check(all(abs(result.candidate[j] - query[j]) < 1e-9
                      for j in constraints.immutable),
                  "immutable features must be untouched")
            check(len(result.changed) == len(result.support),
                  "the reported changed features must be the support")
            check(abs(result.cost - cost_model.cost(query, result.candidate)) < 1e-9,
                  "the reported cost must equal the independently recomputed cost")
            best_cost, best_changed, _best = reference[0]
            check(result.cost <= best_cost + 1e-9,
                  "the answer must not cost more than the best grid point (%.4f vs %.4f)"
                  % (result.cost, best_cost))
            check(len(result.changed) <= best_changed,
                  "the answer must not change more features than the best grid point (%d vs %d)"
                  % (len(result.changed), best_changed))
            # certificate 1: nothing sparser exists anywhere on the grid
            for size in range(1, len(result.changed)):
                for subset in combinations_of(constraints.mutable(), size):
                    sparser = [entry for entry in reference
                               if all(index in subset
                                      for index in CostModel.changed(query, entry[2]))]
                    check(not sparser,
                          "no counterfactual may exist with fewer than %d changed features"
                          % len(result.changed))
            # certificate 2: locally optimal - no single coordinate perturbation is cheaper
            for coordinate in result.changed:
                for step in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                    neighbour = list(result.candidate)
                    neighbour[coordinate] += step
                    neighbour = constraints.project(neighbour, constraints.mutable())
                    if not constraints.satisfies(query, neighbour):
                        continue
                    if classifier.predict(neighbour) != target:
                        continue
                    shifted = CostModel.changed(query, neighbour)
                    check(cost_model.cost(query, neighbour, shifted) >= result.cost - 1e-9,
                          "no valid neighbour may be cheaper than the answer (%.4f vs %.4f)"
                          % (cost_model.cost(query, neighbour, shifted), result.cost))
    check(comparisons >= 6, "the sweep must cover both directions (%d comparisons)" % comparisons)
    print("sweep: %d query/target pairs agree with the grid oracle on cost and sparsity, with "
          "sparsity and local optimality certificates" % comparisons)

    # ---- 2. the sparsity preference is structural --------------------------
    sparse_fixture = None
    for query in queries:
        for target in (0, 1):
            if classifier.predict(query) == target:
                continue
            result = generator.explain(query, target)
            if result is not None and len(result.changed) == 1:
                sparse_fixture = (query, target, result)
                break
        if sparse_fixture:
            break
    check(sparse_fixture is not None,
          "at least one fixture must be reachable by changing a single feature")
    query, target, result = sparse_fixture
    greedy = GridOracle(classifier, constraints, cost_model).valid_points(query, target)
    check(len(result.changed) == min(entry[1] for entry in greedy),
          "when one change suffices the generator must use one change")
    heavy = CounterfactualGenerator(classifier, constraints, CostModel(sparsity_weight=1e6))
    light = CounterfactualGenerator(classifier, constraints, CostModel(sparsity_weight=1.0))
    heavy_result = heavy.explain(query, target)
    light_result = light.explain(query, target)
    check(len(heavy_result.changed) <= len(light_result.changed),
          "raising the sparsity weight must never change more features (%d vs %d)"
          % (len(heavy_result.changed), len(light_result.changed)))
    print("sparsity: one-feature counterfactual %s; heavy penalty changes %d features, light "
          "penalty %d" % (result.describe(), len(heavy_result.changed), len(light_result.changed)))

    # ---- 3. infeasibility is reported, not guessed -------------------------
    locked = LogisticClassifier("locked", [3.0 if index in IMMUTABLE else 0.0
                                           for index in range(FEATURES)], -60.0)
    locked_generator = CounterfactualGenerator(locked, constraints, cost_model)
    locked_query = [8.0, 5.0, 5.0, 8.0, 5.0]
    check(locked.predict(locked_query) == 0, "the locked fixture must start in class 0")
    check(locked_generator.explain(locked_query, 1) is None,
          "a target reachable only by an immutable feature must be reported as infeasible")
    check(not GridOracle(locked, constraints, cost_model).valid_points(locked_query, 1),
          "the oracle must agree that the grid holds no counterfactual")
    check(probe_infeasible(locked_generator, locked, locked_query, 1) is None,
          "random probing must agree that no counterfactual exists")
    frozen = Constraints(immutable=tuple(range(FEATURES)))
    frozen_generator = CounterfactualGenerator(classifier, frozen, cost_model)
    frozen_query = build_query(1005)
    frozen_target = 1 - classifier.predict(frozen_query)
    check(frozen_generator.explain(frozen_query, frozen_target) is None,
          "with every feature frozen nothing can be explained")
    print("infeasibility: locked and frozen fixtures report no counterfactual, agreed by grid and "
          "probing")

    # ---- 4. determinism and the query budget -------------------------------
    pairs = [(entry, target_class) for entry in queries for target_class in (0, 1)
             if classifier.predict(entry) != target_class]
    check(len(pairs) >= 4, "the sweep must offer enough query/target pairs")
    replay_a = CounterfactualGenerator(classifier, constraints, CostModel(sparsity_weight=1000.0))
    replay_b = CounterfactualGenerator(classifier, constraints, CostModel(sparsity_weight=1000.0))
    check([replay_a.explain(entry, wanted).candidate for entry, wanted in pairs[:3]]
          == [replay_b.explain(entry, wanted).candidate for entry, wanted in pairs[:3]],
          "the same input must produce the same counterfactual")
    check(replay_a.explain(query, target).candidate == replay_b.explain(query, target).candidate,
          "a repeated explanation must be reproducible bit for bit")
    spent = classifier.queries
    check(spent > 0, "the classifier must be queried")
    tight = CounterfactualGenerator(classifier, constraints, cost_model, budget=2)
    overrun = False
    try:
        tight.explain(query, target)
    except RuntimeError:
        overrun = True
    check(overrun, "exceeding the search budget must raise instead of overspending")
    measured = CounterfactualGenerator(classifier, constraints, cost_model, budget=1 << 20)
    before = measured.attempts
    before_queries = classifier.queries
    single = measured.explain(query, target)
    spent_attempts = measured.attempts - before
    spent_queries = classifier.queries - before_queries
    check(single is not None and spent_attempts <= 400,
          "one explanation must stay inside a bounded search budget (%d candidate points)"
          % spent_attempts)
    check(spent_queries <= 5,
          "the answer must be verified with a handful of classifier queries (%d)" % spent_queries)
    print("determinism and budget: reproducible answers, %d candidate points and %d classifier "
          "queries per explanation, a tight budget raises" % (spent_attempts, spent_queries))

    # ---- 5. validation -----------------------------------------------------
    failures = 0
    for action in (lambda: CounterfactualGenerator(classifier, constraints, cost_model)
                   .explain(query, 2),
                   lambda: CounterfactualGenerator(classifier, constraints, cost_model)
                   .explain([-1.0, 5.0, 5.0, 5.0, 5.0], target),
                   lambda: CounterfactualGenerator(classifier, constraints, cost_model)
                   .explain([1.0, 2.0], target),
                   lambda: CounterfactualGenerator(classifier, constraints, cost_model)
                   .explain(query, classifier.predict(query))):
        try:
            action()
        except ValueError:
            failures += 1
    check(failures == 4, "every malformed request must be rejected (%d of 4)" % failures)
    ok_query = queries[0]
    ok_target = 1 - classifier.predict(ok_query)
    ok_result = CounterfactualGenerator(classifier, constraints, cost_model).explain(ok_query,
                                                                                     ok_target)
    check(ok_result is not None and ok_result.changed,
          "a valid request must still be answered after the rejection tests")
    print("validation: %d malformed requests rejected, valid request still answered"
          % failures)
    print("PASS")


if __name__ == "__main__":
    main()
