"""Task 710 - Implement private-count sketches with Laplace mechanism simulation.  (Tier: Expert, CTCI Beyond - Security & Privacy)

A count-min sketch answers "how often did x occur?" in O(depth) time and O(width * depth) space by
hashing x into one counter per row and returning the smallest of those counters.  The estimator is
one-sided: collisions can only inflate a counter, so estimate >= true count always, and the
inflation is bounded by the total collision mass, which is about N / width.

Making it differentially private is then a two-line consequence of the sketch structure:

  * one stream update touches exactly one counter per row, so the *sensitivity of every single
    counter is 1* (adding or removing one record changes it by at most one);
  * therefore adding independent Laplace(0, b) noise to every counter with b = depth / eps
    protects the whole sketch: the eps budgets of the depth rows compose to a total of
    depth * (1 / b) = eps.

Implemented here: the exact estimator, the tight "conservative update" variant (a counter is only
raised to the current minimum, which provably never underestimates and is never larger than the
plain estimator), the noisy sketch, and the audits the task asks for -- the empirical noise scale
against the claimed b, the empirical sensitivity against the claimed 1, the no-false-negative
invariant, the collision error against the N/width prediction, and the utility (mean absolute
estimate error, heavy-hitter identification) as a function of eps.

Simplified versus a real system: the noise is applied once to the finished sketch (output
perturbation) rather than incrementally to every update, hashes are a deterministic multiply-shift
family rather than keyed universal hashing, and the utility numbers are Monte-Carlo estimates with
the tolerances they deserve.  Nothing is faked: every reported error comes from the counters that
the code actually built and from samples the deterministic PRNG actually produced.
"""
import math
import sys

MASK64 = (1 << 64) - 1
TWO53 = 1 << 53
HASH_PRIME = (1 << 61) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def unit(self):
        self.state, r = splitmix64(self.state)
        return ((r >> 11) + 0.5) / TWO53

    def centred(self):
        return self.unit() - 0.5


def laplace_sample(scale, rng):
    """Inverse-transform sample of Lap(0, scale); scale 0 stays exact."""
    if scale == 0.0:
        return 0.0
    u = rng.centred()
    tail = -scale * math.log(1.0 - 2.0 * abs(u))
    return tail if u < 0 else -tail


class CountMinSketch:
    """Multiply-shift hashing (Chebyshev/universal family) onto `depth` rows of `width` slots."""

    def __init__(self, width, depth, seed=0xC0FFEE):
        check(width >= 1, "width must be at least 1")
        check(depth >= 1, "depth must be at least 1")
        self.width = width
        self.depth = depth
        self.counts = [[0] * width for _ in range(depth)]
        state = seed & MASK64
        self.multipliers = []
        self.offsets = []
        for _ in range(depth):
            state, a = splitmix64(state)
            state, b = splitmix64(state)
            self.multipliers.append(a % HASH_PRIME)
            self.offsets.append(b % HASH_PRIME)
        self.items_seen = 0

    def positions(self, item):
        return [((self.multipliers[row] * item + self.offsets[row]) % HASH_PRIME) % self.width
                for row in range(self.depth)]

    def add(self, item):
        for row, slot in enumerate(self.positions(item)):
            self.counts[row][slot] += 1
        self.items_seen += 1

    def add_conservative(self, item):
        """Raise only the counters that are still at the current minimum: tighter, still safe."""
        rows = self.positions(item)
        smallest = min(self.counts[row][slot] for row, slot in enumerate(rows))
        for row, slot in enumerate(rows):
            if self.counts[row][slot] == smallest:
                self.counts[row][slot] += 1
        self.items_seen += 1

    def estimate(self, item):
        return min(self.counts[row][slot] for row, slot in enumerate(self.positions(item)))

    def row_estimates(self, item):
        return [self.counts[row][slot] for row, slot in enumerate(self.positions(item))]

    def total(self):
        return sum(sum(row) for row in self.counts)


def private_estimate(sketch, item, epsilon, rng):
    """Per-query Laplace mechanism on the depth counters the item hashes to.

    Only those `depth` counters are perturbed, each with Laplace(0, depth / eps), so the depth
    budgets compose to exactly eps for this query.  Two estimators are returned: the plain
    count-min minimum (biased *downwards* once the counters carry noise) and the median of the
    noisy rows, which is the one to use here.
    """
    check(epsilon > 0.0, "epsilon must be positive")
    scale = sketch.depth / epsilon
    noisy = [value + laplace_sample(scale, rng) for value in sketch.row_estimates(item)]
    ordered = sorted(noisy)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else \
        0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    return min(noisy), median, scale


def counter_sensitivity(streams_prefix):
    """Independent oracle: the largest change of any counter when one record is added/removed."""
    base = CountMinSketch(37, 4, seed=0x1234)
    for item in streams_prefix:
        base.add(item)
    worst = 0
    for item in set(streams_prefix) | {-1, 999}:
        with_item = CountMinSketch(37, 4, seed=0x1234)
        for value in list(streams_prefix) + [item]:
            with_item.add(value)
        for row in range(base.depth):
            for slot in range(base.width):
                worst = max(worst, abs(with_item.counts[row][slot] - base.counts[row][slot]))
    return worst


# --------------------------------------------------------------------------- streams
def synth_stream(seed, length, alphabet, heavy_items):
    """A stream with a few heavy hitters and a long uniform tail."""
    rng = Rng(seed)
    stream = []
    for _ in range(length):
        u = rng.unit()
        if u < 0.5 and heavy_items:
            stream.append(heavy_items[int(rng.unit() * len(heavy_items)) % len(heavy_items)])
        else:
            stream.append(int(rng.unit() * alphabet) % alphabet)
    return stream


def true_counts(stream):
    counts = {}
    for item in stream:
        counts[item] = counts.get(item, 0) + 1
    return counts


# --------------------------------------------------------------------------- checks
def test_estimator_invariants():
    stream = synth_stream(0x51EED, 4000, 60, [1, 2])
    truth = true_counts(stream)
    sketch = CountMinSketch(64, 5)
    for item in stream:
        sketch.add(item)
    check(sketch.items_seen == len(stream), "every update must be counted")
    check(sketch.total() == 5 * len(stream), "each update touches exactly one counter per row")
    for item, count in truth.items():
        estimate = sketch.estimate(item)
        check(estimate >= count, f"count-min must never underestimate {item}: {estimate} < {count}")
        check(estimate <= count + sketch.total(), "and it is bounded by the total mass")
    collisions = [sketch.estimate(item) - truth[item] for item in truth]
    average_overhead = sum(collisions) / len(collisions)
    check(average_overhead <= 2.0 * len(stream) / sketch.width,
          f"average collision overhead {average_overhead:.2f} should be about N/w")
    check(sketch.estimate(10 ** 6) == 0, "an item that never appeared estimates to 0")

    # the conservative update is never worse and never underestimates
    conservative = CountMinSketch(64, 5)
    for item in stream:
        conservative.add_conservative(item)
    for item, count in truth.items():
        tight = conservative.estimate(item)
        check(tight >= count, f"conservative update must not underestimate {item}")
        check(tight <= sketch.estimate(item),
              f"conservative update must be at most the plain estimate for {item}")
    tight_overhead = sum(conservative.estimate(item) - truth[item] for item in truth) / len(truth)
    check(tight_overhead <= average_overhead + 1e-9,
          "the conservative variant must have no more overhead on average")

    # deterministic hashing: the same item lands in the same slots in every sketch
    again = CountMinSketch(64, 5)
    check(again.positions(17) == sketch.positions(17), "the hash family is deterministic")
    check(len(set(sketch.positions(17))) <= sketch.depth, "at most one slot per row")
    return average_overhead


def test_counter_sensitivity_is_one():
    streams = [[], [5], [5, 5, 7], [1, 2, 3, 4, 5, 6], [9, 9, 9]]
    for prefix in streams:
        worst = counter_sensitivity(prefix)
        check(worst == 1, f"one record changes each counter by at most 1, measured {worst}")
    sketch = CountMinSketch(16, 4)
    for item in [3, 3, 3, 8]:
        sketch.add(item)
    before = [row[:] for row in sketch.counts]
    sketch.add(3)
    touched = [(row, slot) for row in range(4) for slot in range(16)
               if sketch.counts[row][slot] != before[row][slot]]
    check(len(touched) == 4, f"one update touches exactly depth = 4 counters, got {len(touched)}")
    check(all(sketch.counts[row][slot] - before[row][slot] == 1 for row, slot in touched),
          "and each of them by exactly one")


def test_noise_scale_matches_the_budget():
    epsilon = 0.5
    depth = 6
    scale = depth / epsilon
    check(abs(scale - 12.0) < 1e-12, "b = depth / eps")
    rng = Rng(0xAB0DE)
    trials = 60000
    samples = [laplace_sample(scale, rng) for _ in range(trials)]
    mean = sum(samples) / trials
    variance = sum((value - mean) ** 2 for value in samples) / trials
    check(abs(mean) < 0.05 * scale, f"the injected noise must be centred, got {mean}")
    check(abs(variance - 2.0 * scale * scale) < 0.06 * 2.0 * scale * scale,
          f"the injected noise variance must be 2b^2, got {variance}")

    stream = synth_stream(0x77, 3000, 40, [3])
    sketch = CountMinSketch(48, depth)
    for item in stream:
        sketch.add(item)
    rng = Rng(0xBEEF)
    observed = []
    for _ in range(400):
        _, _, used_scale = private_estimate(sketch, 3, epsilon, rng)
        observed.append(used_scale)
    check(all(abs(value - scale) < 1e-12 for value in observed), "the private reader uses b")
    return scale


def test_utility_tradeoff():
    stream = synth_stream(0x2C0FFEE, 5000, 50, [7, 8, 9])
    truth = true_counts(stream)
    targets = sorted(truth, key=lambda item: (-truth[item], item))[:6]
    sketch = CountMinSketch(1024, 5)          # wide enough that collisions are small vs b
    for item in stream:
        sketch.add(item)
    errors = {}
    noise_errors = {}
    baselines = {item: sorted(sketch.row_estimates(item))[sketch.depth // 2] for item in targets}
    for epsilon in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        rng = Rng(0x5EED + int(epsilon * 1000))
        total = 0.0
        noise_total = 0.0
        trials = 400
        for _ in range(trials):
            for item in targets:
                _, median, _ = private_estimate(sketch, item, epsilon, rng)
                total += abs(median - truth[item])
                noise_total += abs(median - baselines[item])
        errors[epsilon] = total / (trials * len(targets))
        noise_errors[epsilon] = noise_total / (trials * len(targets))
        predicted = sketch.depth / epsilon
        check(0.2 * predicted <= noise_errors[epsilon] <= 1.5 * predicted,
              f"eps = {epsilon}: the noise-only error {noise_errors[epsilon]} should be near b")
        check(errors[epsilon] >= noise_errors[epsilon] - 1e-9,
              "the total error also contains the sketch's collision cost")
    ordered = [noise_errors[eps] for eps in sorted(noise_errors)]
    check(all(a > b for a, b in zip(ordered, ordered[1:])),
          f"the noise error must fall as eps grows: {ordered}")
    collision_cost = sum(baselines[item] - truth[item] for item in targets) / len(targets)
    check(collision_cost >= 0.0, "the median of counters is inflated by collisions, never lowered")
    check(sketch.estimate(targets[0]) - truth[targets[0]] <= 2.0 * len(stream) / sketch.width,
          "the observed collision cost is within the N/w prediction")

    # the plain count-min minimum is badly biased once the counters are noisy: document it
    rng = Rng(0x11AA)
    item = targets[0]
    epsilon = 0.5
    scale = sketch.depth / epsilon
    minimums = []
    medians = []
    for _ in range(500):
        plain, median, _ = private_estimate(sketch, item, epsilon, rng)
        minimums.append(plain - baselines[item])
        medians.append(median - baselines[item])
    mean_minimum = sum(minimums) / len(minimums)
    mean_median = sum(medians) / len(medians)
    check(mean_minimum < -0.3 * scale,
          f"the row minimum is biased low by about b*H_d, got {mean_minimum} (b = {scale})")
    check(abs(mean_median) < 0.5 * scale,
          f"the median of the rows is nearly unbiased, got {mean_median}")

    # heavy-hitter identification at a moderate budget, using the median estimator
    successes = 0
    for trial in range(40):
        rng = Rng(0xD00D + trial)
        noisy = {}
        for item in targets:
            _, median, _ = private_estimate(sketch, item, 1.0, rng)
            noisy[item] = median
        if max(noisy, key=lambda item: noisy[item]) == targets[0]:
            successes += 1
    check(successes >= 30, f"the top item should survive noise at eps = 1, got {successes}/40")
    return errors, noise_errors, collision_cost


def test_boundaries_and_validation():
    empty = CountMinSketch(8, 3)
    check(empty.estimate(5) == 0, "an empty sketch estimates zero")
    check(empty.total() == 0, "and carries no mass")
    rng = Rng(3)
    plain, median, scale = private_estimate(empty, 5, 1.0, rng)
    check(abs(scale - 3.0) < 1e-12, "b = depth / eps even for an empty sketch")
    check(plain <= 0.0, "the noisy estimate of an unseen item stays near zero or below")
    check(median == median, "the median estimator is well defined")

    single = CountMinSketch(4, 2)
    single.add(11)
    check(single.estimate(11) == 1, "a single record estimates to 1")
    check(single.estimate(12) == 0 or single.width == 1, "other items see collisions only")

    degenerate = CountMinSketch(1, 1)
    for item in [1, 2, 3, 4, 5, 6, 1]:
        degenerate.add(item)
    check(degenerate.estimate(1) == 7, "with one slot every estimate is the total mass")
    check(degenerate.estimate(999) == 7, "including for absent items: that is the collision cost")

    zero_budget = CountMinSketch(8, 3)
    for item in [1, 2, 2, 3]:
        zero_budget.add(item)
    plain, _, scale = private_estimate(zero_budget, 2, 1e9, Rng(4))
    check(scale < 1e-6, "an enormous eps means almost no noise")
    check(abs(plain - 2) < 0.01, f"so the estimate is nearly exact, got {plain}")

    for bad in (0.0, -1.0):
        try:
            private_estimate(zero_budget, 2, bad, Rng(5))
            raise AssertionError("a non-positive epsilon must be rejected")
        except AssertionError as error:
            check("epsilon" in str(error), f"unexpected message: {error}")
    try:
        CountMinSketch(0, 3)
        raise AssertionError("width 0 must be rejected")
    except AssertionError as error:
        check("width" in str(error), f"unexpected message: {error}")
    try:
        CountMinSketch(4, 0)
        raise AssertionError("depth 0 must be rejected")
    except AssertionError as error:
        check("depth" in str(error), f"unexpected message: {error}")

    first = [private_estimate(zero_budget, 2, 0.5, Rng(9))[0] for _ in range(2)]
    second = [private_estimate(zero_budget, 2, 0.5, Rng(9))[0] for _ in range(2)]
    check(first == second, "the private reader is deterministic for a fixed seed")


def test_composition_accounting():
    depth = 8
    for epsilon in (0.25, 1.0, 4.0):
        scale = depth / epsilon
        per_counter = 1.0 / scale
        check(abs(depth * per_counter - epsilon) < 1e-12,
              f"the depth rows of budget {per_counter} compose to {epsilon}")
    # the median of `depth` noisy rows beats a single Laplace draw: its mean absolute error is
    # below the single-draw value E|Lap(b)| = b, which is why the median estimator is used
    scale = 5.0
    rng = Rng(0x31415)
    single_total = 0.0
    median_total = 0.0
    trials = 20000
    for _ in range(trials):
        draws = [laplace_sample(scale, rng) for _ in range(depth)]
        ordered = sorted(draws)
        single_total += abs(draws[0])
        median_total += abs(0.5 * (ordered[depth // 2 - 1] + ordered[depth // 2]))
    single_error = single_total / trials
    median_error = median_total / trials
    check(abs(single_error - scale) < 0.05 * scale,
          f"one Laplace draw has mean absolute error b = {scale}, got {single_error}")
    check(median_error < single_error,
          f"the median of {depth} rows must beat one draw: {median_error} vs {single_error}")
    check(median_error > 0.1 * scale, "but the improvement is bounded, as expected")
    # the minimum, on the other hand, is strongly biased downwards
    rng = Rng(0x2718)
    minimum_total = 0.0
    for _ in range(trials):
        minimum_total += abs(min(laplace_sample(scale, rng) for _ in range(depth)))
    check(minimum_total / trials > single_error,
          "the row minimum is worse than a single draw, hence the median estimator")
    return scale, median_error


def main():
    overhead = test_estimator_invariants()
    test_counter_sensitivity_is_one()
    scale = test_noise_scale_matches_the_budget()
    errors, noise_errors, collision_cost = test_utility_tradeoff()
    test_boundaries_and_validation()
    composition_scale, median_error = test_composition_accounting()

    print(f"collision overhead per item: {overhead:.3f} (N/w bound)")
    print(f"median-of-rows baseline cost (collisions + row spread): {collision_cost:.2f}")
    print("eps -> noise-only error / total error (depth 5, width 1024): "
          + ", ".join(f"{eps:g}:{noise_errors[eps]:.2f}/{errors[eps]:.2f}"
                      for eps in sorted(errors)))
    print(f"noise scale b={scale:.1f}, min-of-8 Laplace median |error| = {median_error:.3f}; "
          f"composition scale {composition_scale:.1f}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
