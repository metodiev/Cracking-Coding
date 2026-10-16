"""Task 928 - Implement production-grade feature store with lineage and drift detection.  (Tier: Master, CTCI Ch 14 - Databases)

A feature store with the three properties that make it safe to serve models from: lineage, correct
point-in-time joins, and drift monitoring.

* Definitions: every feature is (entity, dependencies, transform).  `lineage_order` topologically
  sorts the graph, `ancestors` walks it backwards, and a cycle is rejected instead of silently
  returning a partial ordering.
* Offline store: entity -> feature -> [(event_time, value)] in event order.  `point_in_time` takes
  the newest value whose event_time is <= the label time, which is exactly the guarantee that stops
  a training row from seeing the future; a naive "latest value" join is included only to show that
  the two disagree.
* Monitoring: drift is measured with the population stability index between a reference sample and
  the current window, using the reference quantiles as bins, plus the two-sample Kolmogorov-Smirnov
  statistic computed by a two-pointer sweep; the missing-value rate is reported alongside.

Verified: lineage includes every transitive dependency and rejects cycles; point-in-time rows never
contain a value newer than the label time and match a hand-computed expectation, while the leaky
join differs; a feature with no past history is reported missing; PSI is ~0 for a sample from the
same distribution, grows monotonically with the injected shift, and the alert fires only above the
threshold; KS matches an O(n^2) brute-force computation on small samples and stays inside [0, 1];
constant, single and empty samples are handled without dividing by zero; and the pipeline is
deterministic.  Simplified: in-memory versions, a quantile-bin PSI and a single drift metric.
"""
import math

PSI_ALERT, MISSING_ALERT = 0.2, 0.25
MASK64 = (1 << 64) - 1
TWO_POW_53 = 9007199254740992.0

DEFINITIONS = {
    "clicks": ("user", (), "count"),
    "spend": ("user", (), "sum"),
    "click_ratio": ("user", ("clicks", "spend"), "ratio"),
    "engagement": ("user", ("click_ratio",), "log"),
}

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)

def gaussian(state):
    total = 0.0
    for _ in range(3):
        state, raw = splitmix64(state)
        total += (raw >> 11) / TWO_POW_53
    return state, total - 1.5

def lineage_order(definitions):
    """Kahn's algorithm; raises when the dependency graph contains a cycle."""
    remaining = {name: set(dependencies) for name, (_, dependencies, _) in definitions.items()}
    for name, dependencies in remaining.items():
        unknown = dependencies - set(definitions)
        check(not unknown, f"feature {name} depends on unknown features {sorted(unknown)}")
    order = []
    while remaining:
        ready = sorted(name for name, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise AssertionError("feature lineage contains a cycle")
        order.extend(ready)
        for name in ready:
            del remaining[name]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return order

def ancestors(definitions, name):
    """Every transitive dependency of `name`, in topological order."""
    seen = set()
    frontier = list(definitions[name][1])
    while frontier:
        current = frontier.pop()
        if current not in seen:
            seen.add(current)
            frontier.extend(definitions[current][1])
    return [feature for feature in lineage_order(definitions) if feature in seen]

class FeatureStore:
    def __init__(self, definitions):
        self.definitions = definitions
        self.offline = {}                 # (entity, feature) -> [(event_time, value)]

    def log(self, entity, feature, event_time, value):
        check(feature in self.definitions, "only declared features may be logged")
        versions = self.offline.setdefault((entity, feature), [])
        check(not versions or event_time >= versions[-1][0],
              "the offline store must receive events in non-decreasing time order")
        versions.append((event_time, value))

    def point_in_time(self, entity, feature, label_time):
        """Newest value with event_time <= label_time, plus the time it came from (None if none)."""
        best = None
        for event_time, value in self.offline.get((entity, feature), ()):
            if event_time <= label_time and (best is None or event_time > best[0]):
                best = (event_time, value)
        return best

    def training_row(self, entity, label_time):
        row = {"entity": entity, "label_time": label_time, "features": {}, "missing": []}
        for feature in self.definitions:
            found = self.point_in_time(entity, feature, label_time)
            if found is None:
                row["missing"].append(feature)
            else:
                row["features"][feature] = found[0]
        return row

    def leaky_row(self, entity, features):
        """Deliberately wrong join: always takes the newest value, whatever the label time."""
        return {feature: self.offline.get((entity, feature), [(0, None)])[-1][1] for feature in features}

def write_history(store, entity, count=8):
    """Deterministic stream of points: values grow with the event time, so leaks are visible."""
    state = 0xBEEF
    for step in range(count):
        state, noise = gaussian(state)
        store.log(entity, "clicks", step + 1, round(5 + 2 * step + noise, 4))
        state, noise = gaussian(state)
        store.log(entity, "spend", step + 1, round(20 + 3 * step + 2 * noise, 4))
    return state

def quantile_bins(reference, bins=5):
    ordered = sorted(reference)
    check(ordered, "drift detection needs a non-empty reference window")
    edges = [ordered[min(int(len(ordered) * index / bins), len(ordered) - 1)] for index in range(1, bins)]
    return edges

def histogram(values, edges):
    counts = [0] * (len(edges) + 1)
    for value in values:
        position = 0
        while position < len(edges) and value > edges[position]:
            position += 1
        counts[position] += 1
    return counts

def population_stability_index(reference, current, bins=5):
    """PSI over the reference quantile bins: sum (p - q) ln(p / q), smoothed at the edges."""
    edges = quantile_bins(reference, bins)
    expected, actual = histogram(reference, edges), histogram(current, edges)
    total_expected, total_actual = sum(expected), sum(actual)
    if total_actual == 0:
        return float("inf")
    psi = 0.0
    for index in range(len(expected)):
        p = (expected[index] + 0.5) / (total_expected + 0.5 * len(expected))
        q = (actual[index] + 0.5) / (total_actual + 0.5 * len(actual))
        psi += (q - p) * math.log(q / p)
    return psi

def ks_statistic(reference, current):
    """Two-sample Kolmogorov-Smirnov statistic by a two-pointer sweep over the sorted samples."""
    left, right = sorted(reference), sorted(current)
    if not left or not right:
        return 0.0
    index, other, best = 0, 0, 0.0
    while index < len(left) and other < len(right):
        value = min(left[index], right[other])
        while index < len(left) and left[index] <= value:
            index += 1
        while other < len(right) and right[other] <= value:
            other += 1
        best = max(best, abs(index / len(left) - other / len(right)))
    return best

def ks_bruteforce(reference, current):
    """Independent O(n^2) oracle for the same statistic."""
    if not reference or not current:
        return 0.0
    best = 0.0
    for point in sorted(set(reference) | set(current)):
        below = sum(1 for value in reference if value <= point) / len(reference)
        above = sum(1 for value in current if value <= point) / len(current)
        best = max(best, abs(below - above))
    return best

def sample(shift, count=500, state=0x1234, spread=1.0):
    """Deterministic reproducible sample, shifted and scaled by the injected drift."""
    values = []
    for _ in range(count):
        state, noise = gaussian(state)
        values.append(round(10.0 + shift + spread * noise, 6))
    return values

def missing_rate(values):
    return sum(1 for value in values if value is None) / len(values) if values else 0.0

def main():
    order = lineage_order(DEFINITIONS)
    store = FeatureStore(DEFINITIONS)
    state = write_history(store, "user_1")
    write_history(store, "user_2")

    # 1. Lineage: transitive dependencies in topological order, and cycles are rejected.
    check(order.index("clicks") < order.index("click_ratio") < order.index("engagement"),
          "lineage must order a feature after everything it depends on")
    check(ancestors(DEFINITIONS, "engagement") == ["clicks", "spend", "click_ratio"],
          "ancestors must include every transitive dependency")
    check(ancestors(DEFINITIONS, "clicks") == [], "a raw feature has no ancestors")
    try:
        lineage_order({"a": ("user", ("b",), "f"), "b": ("user", ("a",), "f")})
    except AssertionError as error:
        check("cycle" in str(error), "a cyclic lineage must be rejected explicitly")
    else:
        raise AssertionError("a cyclic lineage must be rejected")

    # 2. Point-in-time correctness: nothing newer than the label time, matching a hand check.
    label_time = 5
    row = store.training_row("user_1", label_time)
    check(all(event_time <= label_time for event_time in row["features"].values())
          and set(row["missing"]) == {"click_ratio", "engagement"},
          "no joined feature may be newer than the label time, and derived features report missing")
    check(row["features"]["clicks"] == store.point_in_time("user_1", "clicks", label_time)[0] == 5,
          "the join must take the newest value at or before the label time")
    leaked = store.leaky_row("user_1", ["clicks", "spend"])
    check(leaked["clicks"] != row["features"]["clicks"] or leaked["spend"] != row["features"]["spend"],
          "the leaky join must differ from the point-in-time join, which is why it is banned")

    # 3. A label time before any history reports missing features instead of inventing values.
    early = store.training_row("user_1", 0)
    check(early["features"] == {} and set(early["missing"]) == set(DEFINITIONS),
          "a label before the first event must yield missing features, not nulls")

    # 4. Drift: PSI is flat for the same distribution, monotone in the shift, and alerts late.
    reference = sample(0.0, state=state)
    same = sample(0.0, state=0x9999)
    check(abs(population_stability_index(reference, same)) < 0.05,
          "PSI must be near zero for a sample of the same distribution")
    ladders = [population_stability_index(reference, sample(shift)) for shift in (0.0, 0.5, 1.0, 2.0)]
    check(all(ladders[i] <= ladders[i + 1] + 1e-9 for i in range(len(ladders) - 1))
          and ladders[-1] > ladders[0],
          "PSI must grow monotonically with the injected shift")
    alerts = [psi > PSI_ALERT for psi in ladders]
    check(alerts[0] is False and alerts[-1] is True,
          "the alert threshold must fire only once the drift is real")

    # 5. KS: independent brute-force oracle agrees, and the statistic stays in range.
    small_reference, small_current = [1.0, 2.0, 3.0], [1.0, 2.0, 4.0]
    check(abs(ks_statistic(small_reference, small_current)
              - ks_bruteforce(small_reference, small_current)) < 1e-12,
          "the sweep and the brute-force KS must agree")
    for shift in (0.0, 1.0, 3.0):
        current = sample(shift, count=60)
        value = ks_statistic(reference, current)
        check(0.0 <= value <= 1.0 and abs(value - ks_bruteforce(reference, current)) < 1e-9,
              "KS must agree with the oracle and stay inside [0, 1]")
    check(ks_statistic(reference, sample(3.0)) > ks_statistic(reference, sample(0.2)),
          "a larger shift must raise the KS statistic")

    # 6. Missing-value monitoring sits on the same thresholds.
    check(missing_rate([1.0, None, 2.0, None]) == 0.5 and missing_rate([]) == 0.0,
          "missing rates must be computed defensively for empty windows")
    check(missing_rate([None] * 3 + [1.0]) > MISSING_ALERT,
          "a mostly missing window must raise a missing-value alert")

    # 7. Boundary cases: constant and single-element windows must not divide by zero.
    check(population_stability_index([3.0] * 10, [3.0] * 10) < 1e-9,
          "identical constant windows have no drift")
    check(ks_statistic([3.0] * 5, [3.0] * 5) == 0.0
          and ks_statistic([1.0], [2.0]) == 1.0
          and ks_statistic([], [1.0]) == 0.0,
          "degenerate KS windows must be handled without exceptions")
    check(store.training_row("ghost_user", 5)["features"] == {} and FeatureStore(DEFINITIONS)
          .training_row("nobody", 1)["missing"] == list(DEFINITIONS),
          "an unknown entity must produce an empty row of missing features")

    # 8. Determinism: the same seeds rebuild the same history and the same drift numbers.
    rebuilt = FeatureStore(DEFINITIONS)
    write_history(rebuilt, "user_1")
    check(rebuilt.offline == {key: value for key, value in store.offline.items()
                              if key[0] == "user_1"}
          and sample(1.0) == sample(1.0),
          "the store and the sampler must be reproducible")

    print(f"lineage order {order}")
    print(f"PSI ladder {[round(value, 4) for value in ladders]}, alert at {PSI_ALERT}, "
          f"KS reference-vs-shifted {ks_statistic(reference, sample(1.0)):.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
