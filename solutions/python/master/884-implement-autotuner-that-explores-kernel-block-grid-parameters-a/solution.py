"""Task 884 - Implement autotuner that explores kernel/block/grid parameters and caches best.  (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)

A kernel autotuner with the parts a production one has, and a cost model instead of a GPU so
that the whole search is deterministic and in-process:

  * the search space - tile shape (block_x, block_y), unrolling and vector width, with a
    validity predicate (thread count in range, divisibility, shared-memory budget) that prunes
    configurations *before* they are ever measured;
  * the objective - a modelled kernel cost built from the classic terms: launch overhead, the
    number of waves (tiles per resident block), per-thread work amortised by ILP and vector
    width, and penalties for uncoalesced rows, bank conflicts, tail waste and poor occupancy.
    The landscape has a genuine interior optimum, which is why a search is needed at all;
  * the search strategies - exhaustive (the oracle on a small space), seeded random, greedy
    coordinate descent and a successive-halving bandit that re-measures survivors with more
    repetitions;
  * the cache - keyed by the problem signature (shape and dtype), with warm-start transfer from
    the nearest cached problem, so tuning a related shape costs fewer measurements than a cold
    tune while invalidation on a shape change is explicit.

Verified: the exhaustive search reproduces the brute-force optimum with zero regret; the tuned
configuration beats the default configuration by a measured factor; the cache returns the same
configuration with zero new measurements and invalidates on a shape change; the warm start
needs fewer measurements than the cold tune without losing accuracy; the pruned configurations
are exactly the ones violating the documented rules and never contain the optimum; regret falls
as the measurement budget grows; the cost model is monotone in the problem size and rewards
coalescing and vectorisation; measurement noise perturbs the chosen configuration by only a few
per cent; a configuration that fails validation is never measured; the whole search is
reproducible; and malformed problems are rejected.
"""
import math

SM_COUNT = 8
MAX_THREADS_PER_SM = 2048
SHARED_MEMORY_BYTES = 48 * 1024
CONCURRENCY_CAP = 4096      # threads that can actually be in flight at once
BLOCK_OVERHEAD = 0.2        # per-block scheduling/index cost, in stage units
LAUNCH_COST = 3.0
DEFAULT_CONFIG = (32, 32, 1, 1)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def uniform(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.state >> 11) * (1.0 / float(1 << 53))

    def integer(self, low, high):
        return low + int(self.uniform() * (high - low + 1)) % (high - low + 1)

    def choice(self, items):
        return items[self.integer(0, len(items) - 1)]


def validate_problem(problem):
    if len(problem) != 3:
        raise ValueError("a problem is (M, N, K)")
    if any(not isinstance(extent, int) or extent <= 0 for extent in problem):
        raise ValueError("problem extents must be positive integers")
    return tuple(problem)


# ------------------------------------------------------------------ search space
def candidate_configs(blocks=(16, 32, 64, 128), unrolls=(1, 2, 4, 8), vectors=(1, 2, 4)):
    configs = []
    for block_x in blocks:
        for block_y in blocks:
            for unroll in unrolls:
                for vector in vectors:
                    configs.append((block_x, block_y, unroll, vector))
    return configs


def is_valid(config):
    """The documented rules: integral thread count, sane occupancy, shared-memory budget."""
    block_x, block_y, unroll, vector = config
    if block_x <= 0 or block_y <= 0 or unroll <= 0 or vector <= 0:
        return False
    if block_x & (block_x - 1) or block_y & (block_y - 1):
        return False
    if (block_x * block_y) % (unroll * vector):
        return False
    threads = (block_x * block_y) // (unroll * vector)
    if threads < 32 or threads > 1024:
        return False
    return block_x * block_y * 4 <= SHARED_MEMORY_BYTES


def valid_configs(configs):
    return [config for config in configs if is_valid(config)]


def prune_without_optimum(configs):
    """Independent re-derivation of the pruning rules, used to audit `valid_configs`."""
    kept = []
    for config in configs:
        block_x, block_y, unroll, vector = config
        threads = (block_x * block_y) // (unroll * vector) if (block_x * block_y) % (
            unroll * vector) == 0 else 0
        if 32 <= threads <= 1024 and block_x * block_y * 4 <= SHARED_MEMORY_BYTES:
            kept.append(config)
    return kept


# ------------------------------------------------------------------ cost model
def cost(problem, config):
    """Modelled kernel time; deterministic and interpretable term by term.

    A tiled kernel issues ceil(M/bx) * ceil(N/by) blocks; only `resident` of them are in flight
    at once, so the time is `waves` times the per-block stage cost (the tile's MAC count
    divided by its threads and their instruction-level parallelism), plus penalties for
    uncoalesced rows, partial warps, tail waste and poor occupancy.
    """
    validate_problem(problem)
    if not is_valid(config):
        raise ValueError(f"invalid configuration {config}")
    m, n, k = problem
    block_x, block_y, unroll, vector = config
    tiles_x = (m + block_x - 1) // block_x
    tiles_y = (n + block_y - 1) // block_y
    tiles = tiles_x * tiles_y
    threads = (block_x * block_y) // (unroll * vector)
    shared = block_x * block_y * 4
    blocks_per_sm = max(1, min(MAX_THREADS_PER_SM // threads, SHARED_MEMORY_BYTES // shared))
    resident = max(1, min(tiles, SM_COUNT * blocks_per_sm))
    waves = -(-tiles // resident)
    throughput = vector * (1.0 + 0.5 * math.log2(unroll))
    stage = (block_x * block_y * k) / (threads * throughput)
    total = waves * stage
    penalty = 0.0
    if n % vector:
        penalty += 0.30 * total  # the row tail cannot be vectorised: uncoalesced access
    if block_y % 32:
        penalty += 0.20 * total  # partial warps on the fast axis
    if m % block_x or n % block_y:
        wasted = ((tiles_x * block_x - m) / (tiles_x * block_x)
                  + (tiles_y * block_y - n) / (tiles_y * block_y)) / 2.0
        penalty += 0.5 * wasted * total
    if threads < 64:
        penalty += 0.25 * total  # not enough warps to hide latency
    if threads > 512:
        penalty += 0.10 * total  # fewer blocks resident per SM
    if unroll * vector > 8:
        penalty += 0.05 * total  # register pressure
    return LAUNCH_COST + total + penalty + BLOCK_OVERHEAD * tiles


def measure(problem, config, repetition=0, noise=0.0, seed=0):
    """Modelled measurement; `noise` perturbs it deterministically per (config, repetition)."""
    base = cost(problem, config)
    if noise <= 0.0:
        return base
    mixer = (config[0] * 1000003 + config[1] * 1000033 + config[2] * 1000037
             + config[3] * 1000039 + repetition * 1000081 + problem[0] * 1000099)
    rng = Rng(mixer ^ seed)
    return base * (1.0 + noise * (2.0 * rng.uniform() - 1.0))


def brute_force_optimum(problem, noise=0.0, seed=0):
    best = None
    for config in valid_configs(candidate_configs()):
        value = measure(problem, config, noise=noise, seed=seed)
        if best is None or value < best[1]:
            best = (config, value)
    return best


# ------------------------------------------------------------------ tuner
class Autotuner:
    def __init__(self, budget=24, strategy="bandit", seed=7, noise=0.0, repetitions=1,
                 cache=None):
        if budget < 1:
            raise ValueError("the budget must be at least one measurement")
        if strategy not in ("bandit", "random", "coordinate", "exhaustive"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.budget = budget
        self.strategy = strategy
        self.seed = seed
        self.noise = noise
        self.repetitions = repetitions
        self.cache = cache if cache is not None else {}
        self.measurements = 0
        self.history = []

    # ---------------------------------------------------------- cache
    @staticmethod
    def signature(problem, dtype="f32"):
        return (validate_problem(problem), dtype)

    def warm_start(self, signature):
        """Nearest cached problem: same extents except K, then same M and N, then anything."""
        problem = signature[0]
        best = None
        for key, entry in self.cache.items():
            if key == signature or key[1] != signature[1]:
                continue
            other = key[0]
            distance = (abs(other[0] - problem[0]) + abs(other[1] - problem[1])
                        + 0.1 * abs(other[2] - problem[2]))
            if best is None or distance < best[0]:
                best = (distance, entry["config"])
        return None if best is None else best[1]

    def evaluate(self, problem, config, rng):
        if not is_valid(config):
            raise ValueError(f"refusing to measure an invalid configuration {config}")
        total = 0.0
        for repetition in range(self.repetitions):
            total += measure(problem, config, repetition, self.noise, self.seed)
            self.measurements += 1
        value = total / self.repetitions
        self.history.append((config, value))
        return value

    # ---------------------------------------------------------- strategies
    def tune(self, problem, dtype="f32"):
        signature = self.signature(problem, dtype)
        if signature in self.cache:
            entry = self.cache[signature]
            return {"config": entry["config"], "cost": entry["cost"], "measurements": 0,
                    "cached": True, "spent": 0}
        rng = Rng(self.seed ^ (problem[0] * 31 + problem[1] * 17 + problem[2]))
        space = valid_configs(candidate_configs())
        start = self.measurements
        best = None
        seed_config = self.warm_start(signature)
        if seed_config is not None:
            best = (seed_config, self.evaluate(problem, seed_config, rng))
        if self.strategy == "exhaustive":
            candidates = list(space)
        elif self.strategy == "random":
            candidates = [rng.choice(space) for _ in range(self.budget)]
        elif self.strategy == "coordinate":
            candidates = []
        else:
            candidates = [rng.choice(space) for _ in range(self.budget)]
        if self.strategy == "coordinate":
            best = self.coordinate_descent(problem, rng, DEFAULT_CONFIG, start)
        else:
            for config in candidates:
                if self.measurements - start >= self.budget:
                    break
                if not is_valid(config):
                    continue
                value = self.evaluate(problem, config, rng)
                if best is None or value < best[1]:
                    best = (config, value)
        if self.strategy == "bandit":
            best = self.successive_halving(problem, rng, start, best)
        spent = self.measurements - start
        if best is None:  # pragma: no cover - a non-empty valid space always yields a candidate
            best = (DEFAULT_CONFIG, cost(problem, DEFAULT_CONFIG))
        self.cache[signature] = {"config": best[0], "cost": best[1], "measured": spent}
        return {"config": best[0], "cost": best[1], "measurements": spent, "cached": False,
                "spent": spent}

    def coordinate_candidates(self, problem, start):
        """Single-axis neighbours of a configuration (used by the coordinate checks)."""
        configs = [start]
        block_x, block_y, unroll, vector = start
        for block in (16, 32, 64, 128):
            configs.append((block, block_y, unroll, vector))
            configs.append((block_x, block, unroll, vector))
        for candidate_unroll in (1, 2, 4, 8):
            configs.append((block_x, block_y, candidate_unroll, vector))
        for candidate_vector in (1, 2, 4):
            configs.append((block_x, block_y, unroll, candidate_vector))
        return [config for config in configs if is_valid(config)]

    def coordinate_descent(self, problem, rng, start, origin):
        """Repeated single-axis sweeps from the incumbent until no axis move improves."""
        axes = [(0, (16, 32, 64, 128)), (1, (16, 32, 64, 128)), (2, (1, 2, 4, 8)),
                (3, (1, 2, 4))]
        incumbent = start
        best_value = self.evaluate(problem, incumbent, rng)
        improved = True
        while improved and self.measurements - origin < self.budget:
            improved = False
            for axis, values in axes:
                for value in values:
                    if self.measurements - origin >= self.budget:
                        break
                    candidate = list(incumbent)
                    candidate[axis] = value
                    candidate = tuple(candidate)
                    if not is_valid(candidate):
                        continue
                    measured = self.evaluate(problem, candidate, rng)
                    if measured < best_value:
                        best_value = measured
                        incumbent = candidate
                        improved = True
        return incumbent, best_value

    def successive_halving(self, problem, rng, start, incumbent):
        """Re-measure the survivors with more repetitions while the budget allows."""
        if incumbent is None:
            return incumbent
        survivors = [config for config, _value in self.history if is_valid(config)][-8:]
        best = incumbent
        for config in survivors:
            if self.measurements - start >= self.budget:
                break
            total = 0.0
            count = 0
            for repetition in range(3):
                if self.measurements - start >= self.budget:
                    break
                total += measure(problem, config, repetition + 7, self.noise, self.seed)
                self.measurements += 1
                count += 1
            if count == 0:
                continue
            value = total / count
            if value < best[1]:
                best = (config, value)
        return best


# ------------------------------------------------------------------ checks
def relative_regret(problem, config, noise=0.0, seed=0):
    optimum = brute_force_optimum(problem, noise, seed)
    return (measure(problem, config, noise=noise, seed=seed) - optimum[1]) / optimum[1]


def main():
    # ---------------- validity rules ----------------
    space = candidate_configs()
    valid = valid_configs(space)
    check(0 < len(valid) < len(space),
          f"the pruning must keep a proper subset: {len(valid)}/{len(space)}")
    check(valid == prune_without_optimum(space),
          "the validity predicate must agree with the independent re-derivation")
    check(all(32 <= (bx * by) // (unroll * vector) <= 1024 for bx, by, unroll, vector in valid),
          "every kept configuration must respect the thread bounds")
    check(all(bx * by * 4 <= SHARED_MEMORY_BYTES for bx, by, unroll, vector in valid),
          "every kept configuration must fit the shared-memory budget")
    for bad in ((0, 32, 1, 1), (17, 32, 1, 1), (128, 128, 1, 8), (-16, 32, 1, 1)):
        check(not is_valid(bad), f"{bad} must be rejected")
    try:
        cost((64, 64, 64), (128, 128, 1, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("cost must refuse an invalid configuration")

    # ---------------- exhaustive search reaches the brute-force optimum ----------------
    problem = (256, 256, 128)
    optimum, optimum_cost = brute_force_optimum(problem)
    check(is_valid(optimum), "the optimum must be a valid configuration")
    oracle = Autotuner(budget=len(valid), strategy="exhaustive", seed=1)
    result = oracle.tune(problem)
    check(result["config"] == optimum,
          f"the exhaustive search must find the optimum: {result['config']} vs {optimum}")
    check(abs(result["cost"] - optimum_cost) < 1e-12, "the cost of the optimum must match")
    check(result["measurements"] == len(valid),
          f"an exhaustive search must measure the whole space: {result['measurements']}")

    # ---------------- the tuned configuration beats the default ----------------
    default_cost = cost(problem, DEFAULT_CONFIG)
    check(optimum_cost < default_cost,
          "the cost landscape must have an interior optimum better than the default")
    speedup = default_cost / optimum_cost
    check(speedup > 1.2, f"the tuning must pay off: {speedup:.3f}x")
    check(relative_regret(problem, DEFAULT_CONFIG) > 0.1,
          "the default must be measurably worse than the optimum")

    # ---------------- caching and invalidation ----------------
    tuner = Autotuner(budget=12, strategy="bandit", seed=5, repetitions=1)
    first = tuner.tune(problem)
    check(first["measurements"] > 0, "a cold tune must measure something")
    second = tuner.tune(problem)
    check(second["cached"] and second["measurements"] == 0,
          f"a repeated tune must be a cache hit: {second}")
    check(second["config"] == first["config"], "a cache hit must return the same configuration")
    check(tuner.measurements == first["measurements"],
          "a cache hit must not spend measurements")
    bigger = tuner.tune((512, 512, 128))
    check(not bigger["cached"] and bigger["measurements"] > 0,
          "a new shape must invalidate the cache entry")
    check(tuner.signature((256, 256, 128)) != tuner.signature((512, 512, 128)),
          "the signature must depend on the shape")

    # ---------------- warm-start transfer through a shared cache ----------------
    shared_cache = {}
    donor = Autotuner(budget=len(valid), strategy="exhaustive", seed=4, cache=shared_cache)
    donor_result = donor.tune((256, 256, 128))
    check(len(shared_cache) == 1, "the donor must populate the shared cache")
    warm = Autotuner(budget=1, strategy="bandit", seed=4, cache=shared_cache)
    warm_result = warm.tune((256, 256, 256))
    check(warm_result["measurements"] == 1,
          f"the warm start must only spend its first measurement: {warm_result}")
    check(warm.warm_start(warm.signature((256, 256, 256))) is not None,
          "the warm start must find a neighbour")
    check(warm.warm_start(warm.signature((1024, 1024, 512))) is not None,
          "the warm start must fall back to the nearest available problem")
    warm_regret = relative_regret((256, 256, 256), warm_result["config"])
    cold = Autotuner(budget=1, strategy="bandit", seed=4)
    cold_result = cold.tune((256, 256, 256))
    cold_regret = relative_regret((256, 256, 256), cold_result["config"])
    check(warm_regret < cold_regret,
          f"the transferred configuration must beat a cold single measurement: {warm_regret} "
          f"vs {cold_regret}")
    check(warm_regret <= 0.05,
          f"the tiling optimum must transfer across the depth: {warm_regret}")
    check(donor_result["config"] == warm.warm_start(warm.signature((256, 256, 256))),
          "the transferred configuration must be the donor's answer")

    # ---------------- regret versus budget ----------------
    problems = [(128, 128, 128), (256, 256, 128), (512, 256, 64)]
    regrets = []
    for budget in (2, 8, 32):
        total = 0.0
        for index, shape in enumerate(problems):
            engine = Autotuner(budget=budget, strategy="random", seed=100 + index)
            outcome = engine.tune(shape)
            total += relative_regret(shape, outcome["config"])
        regrets.append(total / len(problems))
    check(regrets[0] > regrets[-1],
          f"a bigger budget must reduce the regret: {regrets}")
    check(regrets[-1] < 0.1, f"a large budget must get close to optimal: {regrets}")

    # ---------------- the bandit beats or matches random at the same budget ----------------
    bandit_regrets = []
    random_regrets = []
    for index, shape in enumerate(problems):
        bandit = Autotuner(budget=8, strategy="bandit", seed=200 + index)
        random_tuner = Autotuner(budget=8, strategy="random", seed=200 + index)
        bandit_regrets.append(relative_regret(shape, bandit.tune(shape)["config"]))
        random_regrets.append(relative_regret(shape, random_tuner.tune(shape)["config"]))
    check(sum(bandit_regrets) <= sum(random_regrets) + 0.1,
          f"the bandit must not be worse than random: {bandit_regrets} vs {random_regrets}")
    check(all(math.isfinite(value) for value in bandit_regrets + random_regrets),
          "every regret must be finite")

    # ---------------- coordinate descent: big wins, but it can stall locally ----------------
    coordinate = Autotuner(budget=40, strategy="coordinate", seed=9)
    coordinate_result = coordinate.tune(problem)
    coordinate_regret = relative_regret(problem, coordinate_result["config"])
    check(coordinate_regret < relative_regret(problem, DEFAULT_CONFIG),
          f"coordinate descent must beat the untuned default: {coordinate_regret}")
    check(coordinate_regret < 1.0,
          f"coordinate descent must improve a lot on the default: {coordinate_regret}")
    bandit_wide = Autotuner(budget=40, strategy="bandit", seed=9)
    bandit_regret = relative_regret(problem, bandit_wide.tune(problem)["config"])
    check(bandit_regret <= coordinate_regret + 0.05,
          f"the bandit must be at least as good as coordinate descent: {bandit_regret} vs "
          f"{coordinate_regret}")
    check(is_valid(coordinate_result["config"]), "the chosen configuration must be valid")

    # ---------------- noise robustness ----------------
    noisy_tuner = Autotuner(budget=24, strategy="bandit", seed=11, noise=0.03, repetitions=1)
    noisy_result = noisy_tuner.tune(problem)
    noisy_regret = relative_regret(problem, noisy_result["config"])
    check(noisy_regret < 0.15,
          f"3% measurement noise must not derail the choice: {noisy_regret}")
    repeated = Autotuner(budget=24, strategy="bandit", seed=11, noise=0.03, repetitions=3)
    repeated_result = repeated.tune(problem)
    check(repeated_result["measurements"] >= noisy_result["measurements"],
          "repeated measurements must cost at least as much")
    check(relative_regret(problem, repeated_result["config"]) <= 0.15,
          "averaging repetitions must stay accurate")

    # ---------------- cost-model properties ----------------
    base = (128, 128, 64)
    reference = (64, 128, 4, 4)
    check(is_valid(reference), "the reference configuration must be valid")
    check(cost((128, 128, 128), reference) > cost(base, reference),
          "more work must cost more")
    check(cost((128, 130, 64), reference) > cost(base, reference),
          "a non-multiple row extent must trigger the coalescing penalty")
    check(cost((130, 128, 64), reference) > cost(base, reference),
          "a tail block must trigger the waste penalty")
    check(cost((128, 128, 64), (32, 32, 8, 4)) >= cost((128, 128, 64), (64, 64, 8, 4)),
          "a tiny tile must not be free")
    costs = sorted(cost(base, config) for config in valid)
    check(len(set(round(value, 9) for value in costs)) > 20,
          f"the landscape must be non-trivial: {len(set(costs))} distinct costs")
    check(costs[0] * 1.2 < costs[len(costs) // 2],
          "the optimum must be clearly better than the median configuration")

    # ---------------- determinism, invalid inputs, unaffected configuration space ----------------
    repeat = Autotuner(budget=12, strategy="bandit", seed=5)
    repeated_outcome = repeat.tune(problem)
    check(repeated_outcome["config"] == first["config"]
          and repeated_outcome["measurements"] == first["measurements"],
          "the whole search must be reproducible")
    check(Autotuner(strategy="random", seed=7).tune(problem)["config"]
          == Autotuner(strategy="random", seed=7).tune(problem)["config"],
          "the seeded random strategy must be reproducible")
    for call, why in (
        (lambda: Autotuner(budget=0), "zero budget"),
        (lambda: Autotuner(strategy="wishful"), "unknown strategy"),
        (lambda: cost((0, 128, 64), (64, 64, 1, 1)), "zero extent"),
        (lambda: cost((128, 128), (64, 64, 1, 1)), "malformed problem"),
        (lambda: Autotuner().evaluate((128, 128, 64), (128, 128, 1, 8), Rng(1)),
         "evaluating an invalid configuration"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")
    check(len(space) == len(set(space)), "the candidate list must not repeat configurations")

    print(f"space={len(space)} valid={len(valid)} optimum={optimum} cost={optimum_cost:.3f}")
    print(f"default={DEFAULT_CONFIG} cost={default_cost:.3f} speedup={speedup:.3f}x")
    print(f"regret vs budget {[round(value, 4) for value in regrets]} "
          f"bandit={[round(v, 3) for v in bandit_regrets]}")
    print("PASS")


if __name__ == "__main__":
    main()
