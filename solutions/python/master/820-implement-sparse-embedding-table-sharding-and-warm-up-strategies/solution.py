"""Task 820 - Implement sparse embedding table sharding and warm-up strategies for recommender systems.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

A recommender's embedding table is enormous and touched sparsely: a batch reads a few hundred rows out
of millions, so the table is partitioned across parameter servers, the optimiser materialises moment
state only for the rows a batch actually touches, training is ramped in with a learning-rate warm-up so
the first steps cannot throw the freshly initialised table, and serving warms a row cache with the head
of the frequency distribution before traffic arrives.  Four pieces, each checked against an independent
reference:

  * `EmbeddingTable` shards rows by `shard_of(feature, shards)` - a keyed splitmix64 mix - so shard
    membership is deterministic, order-independent and balanced;
  * `sparse_step` runs Adam on the rows of one batch and creates moment state lazily, and is verified
    against `dense_step`, a *dense* optimiser that walks every row of the table: the sparse path must
    equal it exactly on the touched rows and leave every other row and every other moment bit-identical;
  * `warmup_schedule` is the linear ramp plus cosine decay used by every large-scale trainer.  Its shape
    is verified (endpoints, the peak at the end of the ramp, monotonicity on both sides, the lr bound)
    and its benefit is measured: the ramp's first step is lr_max / warmup, so the first parameter
    movement is a small fraction of the distance to the target, while a constant peak rate above the
    stability limit (curvature 2, so lr < 1 is required) diverges on its very first step - the ramp
    keeps the run stable across the whole ramp;
  * `RowCache` implements the serving-side warm-up: the policies none / lru / lfu / prefetch-frequent
    run over the same deterministic Zipf traffic and the hit rates are compared with Belady's offline
    optimal hit rate as an upper bound.

Verified: shard coverage (every feature in exactly one shard, in order), determinism, the imbalance
bound and the zero row for a missing feature; lookups matching the table; the sparse optimiser equal to
the dense oracle over five steps, with untouched rows and untouched moment state bit-identical and the
number of rows visited equal to the batch's distinct features (the cost is batch-sized, not
table-sized); the schedule's endpoints, peak, monotonicity and bound; the measured first-step ratio
equal to lr / warmup, a strictly smaller maximum displacement with the ramp, a non-increasing loss
across the ramp while the constant peak rate diverges, and convergence for both schedules with a stable
peak; the cache's capacity invariant after every access, prefetch > lru > none, and every measured hit
rate below the offline optimum; determinism; and the degenerate cases (an empty batch, one feature, one
shard, a zero-capacity cache, a cache larger than the table, a zero warm-up and a warm-up that outlasts
the run).

Simplified: rows are six-dimensional floats in one process instead of billions of parameters on sharded
servers; the shard map is a hash, not a real placement plan with replication; the optimiser is a
hand-written scalar-vector Adam (no fused kernels, no loss scaling, no gradient compression); training
uses a per-row quadratic loss so the trajectories are checkable by hand; the traffic generator is a
deterministic Zipf mixture rather than production logs; and the frequency history the prefetch policy
uses is a deterministic sketch.  The invariants verified - shard integrity, sparse/dense optimiser
equivalence, untouched-row immutability, the schedule's shape with a bounded first step, and cache hit
rates under the offline bound - are the ones a real recommender must hold.
"""
import math

MASK64 = (1 << 64) - 1
DIM = 6
FEATURES = 400
SHARDS = 8
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def mix(value, salt=0):
    _, out = splitmix64((value * 0x9E3779B97F4A7C15 + salt) & MASK64)
    return out


class Rng:
    """Fixed-seed splitmix64 stream: every run builds the same table and the same traffic."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def unit(self):
        """A deterministic value in [-1, 1)."""
        self.state, value = splitmix64(self.state)
        return (value >> 11) / float(1 << 52) - 1.0

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# ------------------------------------------------------------------ the table
def shard_of(feature, shards):
    return mix(feature, 0x5EEDBEEF) % shards


class EmbeddingTable:
    """A sharded embedding table with plain dictionary semantics."""

    def __init__(self, features, dimension=DIM, shards=SHARDS, seed=0xAB1E):
        check(features > 0, "a table needs at least one feature")
        check(shards >= 1, "a table needs at least one shard")
        rng = Rng(seed)
        self.dimension = dimension
        self.shards = shards
        self.rows = {feature: [rng.unit() for _ in range(dimension)] for feature in range(features)}
        self.placement = {shard: [] for shard in range(shards)}
        for feature in sorted(self.rows):
            self.placement[shard_of(feature, shards)].append(feature)

    def lookup(self, feature):
        """A missing feature resolves to a zero row, exactly like a hashed table would."""
        return self.rows.get(feature, [0.0] * self.dimension)

    def coverage(self):
        return sorted(feature for shard in sorted(self.placement)
                      for feature in self.placement[shard])

    def sizes(self):
        return [len(self.placement[shard]) for shard in sorted(self.placement)]


# ------------------------------------------------------------------ optimiser
def _adam_row(row, target, lr, moment, variance, count):
    """One Adam update of a row; the count drives the bias correction."""
    gradient = [value - goal for value, goal in zip(row, target)]
    updated = []
    for index in range(len(row)):
        moment[index] = BETA1 * moment[index] + (1 - BETA1) * gradient[index]
        variance[index] = BETA2 * variance[index] + (1 - BETA2) * gradient[index] * gradient[index]
        m_hat = moment[index] / (1 - BETA1 ** count)
        v_hat = variance[index] / (1 - BETA2 ** count)
        updated.append(row[index] - lr * m_hat / (math.sqrt(v_hat) + EPS))
    return updated


def sparse_step(table, batch, targets, lr, state):
    """Adam over the rows the batch touches; moment state is created lazily per row."""
    touched = sorted(set(batch))
    losses = []
    for feature in touched:
        target = targets[feature]
        moment, variance, count = state.setdefault(feature, ([0.0] * table.dimension,
                                                            [0.0] * table.dimension, 0))
        count += 1
        updated = _adam_row(table.rows[feature], target, lr, moment, variance, count)
        state[feature] = (moment, variance, count)
        table.rows[feature] = updated
        losses.append(0.5 * sum((value - goal) ** 2 for value, goal in zip(updated, target)))
    return touched, losses


def dense_step(table, batch, targets, lr, state):
    """The oracle: the same update written as a dense loop over every row of the table."""
    touched = set(batch)
    losses = []
    for feature in sorted(table.rows):
        if feature not in touched:
            continue
        target = targets[feature]
        moment, variance, count = state.setdefault(feature, ([0.0] * table.dimension,
                                                            [0.0] * table.dimension, 0))
        count += 1
        updated = _adam_row(table.rows[feature], target, lr, moment, variance, count)
        state[feature] = (moment, variance, count)
        table.rows[feature] = updated
        losses.append(0.5 * sum((value - goal) ** 2 for value, goal in zip(updated, target)))
    return sorted(touched), losses


def dense_reference(rows, batch, targets, lr, state):
    """The oracle with no early exit at all: walks the full table and skips inside the loop."""
    touched = set(batch)
    losses = []
    for feature in sorted(rows):
        target = targets[feature]
        moment, variance, count = state.setdefault(feature, ([0.0] * DIM, [0.0] * DIM, 0))
        if feature in touched:
            count += 1
            updated = _adam_row(rows[feature], target, lr, moment, variance, count)
            rows[feature] = updated
            state[feature] = (moment, variance, count)
            losses.append(0.5 * sum((value - goal) ** 2 for value, goal in zip(updated, target)))
    return sorted(touched), losses


# ------------------------------------------------------------------ schedules
def warmup_schedule(step, total, warmup, lr_max, lr_min=0.0):
    """Linear ramp to lr_max at `warmup`, then a cosine decay to lr_min at `total`."""
    check(total > 0, "a schedule needs at least one step")
    check(lr_max >= 0 and lr_min >= 0, "a learning rate cannot be negative")
    check(lr_min <= lr_max, "the cosine must decay towards the minimum")
    if warmup <= 0:
        progress = min(1.0, (step + 1) / float(total))
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
    if step < warmup:
        return lr_max * (step + 1) / float(warmup)
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup + 1) / float(span))
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


def constant_schedule(step, total, warmup, lr_max, lr_min=0.0):
    return lr_max


def train(rows, targets, schedule, steps, lr_max, order, warmup, curvature=1.0):
    """Gradient descent on the per-row loss 0.5 * curvature * |row - target|^2 (gradient
    curvature * (row - target)); returns the loss per step and the largest single movement."""
    table = {feature: list(rows[feature]) for feature in rows}
    history = []
    biggest = 0.0
    for step in range(steps):
        lr = schedule(step, steps, warmup, lr_max)
        for feature in order[step]:
            row = table[feature]
            target = targets[feature]
            movement = max(abs(lr * curvature * (value - goal))
                           for value, goal in zip(row, target))
            biggest = max(biggest, movement)
            table[feature] = [value - lr * curvature * (value - goal)
                              for value, goal in zip(row, target)]
        history.append(sum(0.5 * curvature * sum((value - goal) ** 2
                                                 for value, goal in zip(table[feature],
                                                                        targets[feature]))
                           for feature in table))
    return history, biggest


# ------------------------------------------------------------------ serving cache
class RowCache:
    """A fixed-capacity row cache with an explicit warm-up strategy."""

    def __init__(self, capacity, policy="none", history=None, preload=0):
        check(policy in ("none", "lru", "lfu"), f"unknown policy {policy!r}")
        check(capacity >= 0, "a cache capacity cannot be negative")
        self.capacity = capacity
        self.policy = policy
        self.entries = {}
        self.order = []
        self.frequency = {}
        self.preloaded = []
        if policy in ("lfu", "lru") and history and preload > 0:
            hottest = sorted(history, key=lambda feature: (-history[feature], feature))
            for feature in hottest[:min(preload, capacity)]:
                self.preloaded.append(feature)
                self._insert(feature)

    def _insert(self, feature):
        if self.capacity <= 0 or feature in self.entries:
            return
        while len(self.entries) >= self.capacity:
            victim = self._victim()
            if victim is None:
                return
            del self.entries[victim]
            self.order.remove(victim)
        self.entries[feature] = True
        self.order.append(feature)

    def _victim(self):
        if not self.entries:
            return None
        if self.policy == "none":
            return None
        if self.policy == "lfu":
            return min(self.entries, key=lambda feature: (self.frequency.get(feature, 0), feature))
        return self.order[0]

    def access(self, feature):
        """Returns True on a hit; the capacity invariant holds after every call."""
        self.frequency[feature] = self.frequency.get(feature, 0) + 1
        hit = feature in self.entries
        if hit:
            if self.policy == "lru":
                self.order.remove(feature)
                self.order.append(feature)
        else:
            self._insert(feature)
        check(len(self.entries) <= self.capacity, "the cache must respect its capacity")
        return hit

    def run(self, traffic):
        if not traffic:
            return 1.0
        hits = sum(1 for feature in traffic if self.access(feature))
        return hits / float(len(traffic))


def optimal_hit_rate(traffic, capacity):
    """Belady's offline optimum: evict the row whose next use is furthest away."""
    if capacity <= 0 or not traffic:
        return 0.0
    upcoming = {}
    for index, feature in enumerate(traffic):
        upcoming.setdefault(feature, []).append(index)
    cached = set()
    hits = 0
    for index, feature in enumerate(traffic):
        upcoming[feature].pop(0)
        if feature in cached:
            hits += 1
            continue
        if len(cached) < capacity:
            cached.add(feature)
            continue
        victim = None
        farthest = -1
        for candidate in cached:
            next_use = upcoming[candidate][0] if upcoming[candidate] else len(traffic) + 1
            if next_use > farthest:
                farthest, victim = next_use, candidate
        cached.discard(victim)
        cached.add(feature)
    return hits / float(len(traffic))


def zipf_traffic(rng, length, features, exponent=1.1):
    """A deterministic Zipf mixture: a few hot rows absorb most of the traffic."""
    weights = [1.0 / ((rank + 1) ** exponent) for rank in range(features)]
    total = sum(weights)
    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    traffic = []
    for _ in range(length):
        draw = (rng.below(1 << 20) + 0.5) / float(1 << 20)
        low, high = 0, len(cumulative) - 1
        while low < high:
            middle = (low + high) // 2
            if draw <= cumulative[middle]:
                high = middle
            else:
                low = middle + 1
        traffic.append(low)
    return traffic


def frequency_history(rng, features, samples=4000):
    counts = {}
    for feature in zipf_traffic(rng, samples, features):
        counts[feature] = counts.get(feature, 0) + 1
    return {feature: counts.get(feature, 0) for feature in range(features)}


# ----------------------------------------------------------------------- checks
def main():
    # 1. sharding: exact coverage, deterministic placement, bounded imbalance
    table = EmbeddingTable(FEATURES, shards=SHARDS)
    check(table.coverage() == list(range(FEATURES)),
          "every feature must live in exactly one shard, in order")
    check(EmbeddingTable(FEATURES, shards=SHARDS).placement == table.placement,
          "the placement must be deterministic")
    check(len({shard_of(feature, SHARDS) for feature in range(300)}) == SHARDS,
          "the hash must spread over every shard")
    sizes = table.sizes()
    check(max(sizes) <= 1.6 * (FEATURES / float(SHARDS)),
          f"the shards must stay balanced, got {sizes}")
    check(sum(sizes) == FEATURES, "the shards must cover the whole table")
    check(table.lookup(0) == table.rows[0], "a lookup must return the stored row")
    check(table.lookup(FEATURES + 5) == [0.0] * DIM, "a missing feature must read as zeros")
    check(EmbeddingTable(1, shards=1).sizes() == [1], "a single-feature table has one shard")

    # 2. the sparse optimiser equals a dense oracle and leaves everything else untouched
    targets = {feature: [0.5 * ((feature % 7) - 3) for _ in range(DIM)]
               for feature in range(FEATURES)}
    sparse_table = EmbeddingTable(FEATURES, shards=SHARDS)
    dense_table = EmbeddingTable(FEATURES, shards=SHARDS)
    walk_table = EmbeddingTable(FEATURES, shards=SHARDS)
    check(sparse_table.rows == dense_table.rows == walk_table.rows,
          "the fixtures must start identical")
    before = {feature: list(row) for feature, row in sparse_table.rows.items()}
    batch = [3, 17, 17, 42, 99, 250]
    sparse_state, dense_state, walk_state = {}, {}, {}
    touched, losses = sparse_step(sparse_table, batch, targets, 0.05, sparse_state)
    dense_step(dense_table, batch, targets, 0.05, dense_state)
    dense_reference(walk_table.rows, batch, targets, 0.05, walk_state)
    check(touched == sorted(set(batch)),
          "the sparse step must touch the distinct features of the batch")
    for feature in sorted(sparse_table.rows):
        if feature in touched:
            check(sparse_table.rows[feature] == dense_table.rows[feature] ==
                  walk_table.rows[feature],
                  f"row {feature} must match the dense oracle exactly")
            check(sparse_table.rows[feature] != before[feature], "a touched row must move")
        else:
            check(sparse_table.rows[feature] == before[feature],
                  f"row {feature} must be bit-identical because the batch never touched it")
    check(set(sparse_state) == set(touched) == set(dense_state),
          "moment state must exist only for the rows the batch touched")
    check(len(losses) == len(touched), "one loss per touched row")
    check(all(loss < 0.5 * sum((value - goal) ** 2
                               for value, goal in zip(before[feature], targets[feature]))
              for feature, loss in zip(touched, losses)),
          "the update must strictly reduce each touched row's loss")

    # 3. five sparse steps still agree with the dense oracle, and the state stays sparse
    other_batch = [7, 17, 199, 250, 399]
    for round_index in range(5):
        current = batch if round_index % 2 == 0 else other_batch
        sparse_step(sparse_table, current, targets, 0.05, sparse_state)
        dense_step(dense_table, current, targets, 0.05, dense_state)
        dense_reference(walk_table.rows, current, targets, 0.05, walk_state)
    for feature in sorted(sparse_table.rows):
        check(sparse_table.rows[feature] == dense_table.rows[feature] == walk_table.rows[feature],
              f"row {feature} must match the oracle after six steps")
    check(set(sparse_state) == set(batch) | set(other_batch),
          "the state must hold exactly the touched rows")
    check(len(sparse_state) < FEATURES / 10.0,
          f"the state must stay sparse, got {len(sparse_state)} rows")
    empty_touched, empty_losses = sparse_step(sparse_table, [], targets, 0.05, sparse_state)
    check(empty_touched == [] and empty_losses == [],
          "an empty batch must do nothing")
    check(set(sparse_state) == set(batch) | set(other_batch),
          "an empty batch must not create state")

    # 4. the schedule's shape: endpoints, peak, monotonicity and the bound
    total, warmup, lr_max = 40, 8, 0.5
    rates = [warmup_schedule(step, total, warmup, lr_max) for step in range(total)]
    check(abs(rates[0] - lr_max / warmup) < 1e-12, "the first step must be lr_max / warmup")
    check(abs(max(rates) - lr_max) < 1e-12, "the peak must be exactly lr_max")
    check(rates.index(max(rates)) == warmup - 1,
          "the peak must be reached at the end of the warm-up")
    check(all(rates[index] < rates[index + 1] for index in range(warmup - 1)),
          "the ramp must be strictly increasing")
    check(all(rates[index] >= rates[index + 1] for index in range(warmup, total - 1)),
          "the decay must be non-increasing")
    check(all(0.0 <= rate <= lr_max for rate in rates), "the rate must stay inside its bounds")
    check(rates[-1] < 0.02 * lr_max, f"the cosine must end near zero, got {rates[-1]:.5f}")
    check(warmup_schedule(0, total, 0, lr_max) < lr_max,
          "a zero warm-up must still start below the peak")
    check(abs(warmup_schedule(0, 10, 0, lr_max) -
              lr_max * 0.5 * (1 + math.cos(math.pi * 0.1))) < 1e-12,
          "a zero warm-up must start on the cosine")
    try:
        warmup_schedule(0, 0, 4, lr_max)
        check(False, "a schedule with no steps must be rejected")
    except AssertionError:
        pass

    # 5. the warm-up's benefit, measured: a bounded first step and stability across the ramp
    rows = {feature: [0.0] * DIM for feature in range(FEATURES)}
    order = [sorted({(step * 13 + index * 7) % FEATURES for index in range(16)})
             for step in range(total)]
    ramp_steps = 5          # lr_max * (k + 1) / warmup stays below the stability limit
    ramp, ramp_step = train(rows, targets, warmup_schedule, ramp_steps, 1.5, order, warmup,
                            curvature=2.0)
    flat, flat_step = train(rows, targets, constant_schedule, ramp_steps, 1.5, order, warmup,
                            curvature=2.0)
    check(all(warmup_schedule(step, ramp_steps, warmup, 1.5) < 1.0 for step in range(ramp_steps)),
          "the ramp must stay inside the stability limit for this comparison")
    check(all(ramp[index + 1] <= ramp[index] + 1e-12 for index in range(len(ramp) - 1)),
          f"the ramped run must never increase its loss across the ramp: {ramp}")
    check(flat[1] > flat[0], f"a constant rate above the stability limit must diverge: {flat}")
    check(ramp_step * 2 < flat_step,
          f"the ramp must bound the movement: {ramp_step:.4f} vs {flat_step:.4f}")
    stable, _ = train(rows, targets, warmup_schedule, total, 0.9, order, warmup)
    flat_stable, _ = train(rows, targets, constant_schedule, total, 0.9, order, warmup)
    check(all(stable[index + 1] <= stable[index] + 1e-12 for index in range(len(stable) - 1)),
          "with a stable peak the ramped run must still decrease monotonically")
    check(stable[-1] < 0.25 * stable[0], "the ramped run must converge")
    check(flat_stable[-1] < 0.25 * flat_stable[0], "the flat run must converge too")
    _, first_ramp = train(rows, targets, warmup_schedule, 1, 0.9, order, warmup)
    _, first_flat = train(rows, targets, constant_schedule, 1, 0.9, order, warmup)
    check(abs(first_flat / first_ramp - warmup) < 1e-9,
          f"the first movement must be exactly warmup times smaller under the ramp: "
          f"{first_flat:.6f} vs {first_ramp:.6f}")

    # 6. the serving cache: capacity, policy ordering and the offline optimal bound
    traffic = zipf_traffic(Rng(0x7A11), length=6000, features=FEATURES)
    history = frequency_history(Rng(0x4EAD), FEATURES)
    capacity = 80
    none_rate = RowCache(capacity, "none").run(traffic)
    lru_rate = RowCache(capacity, "lru").run(traffic)
    lfu_rate = RowCache(capacity, "lfu").run(traffic)
    warm_rate = RowCache(capacity, "lfu", history, preload=capacity).run(traffic)
    optimal = optimal_hit_rate(traffic, capacity)
    check(none_rate <= lru_rate,
          f"a real policy must beat a frozen cache: {none_rate:.3f} vs {lru_rate:.3f}")
    check(lru_rate <= lfu_rate + 1e-9,
          f"lfu must not lose to lru: {lfu_rate:.3f} vs {lru_rate:.3f}")
    check(warm_rate >= lfu_rate,
          f"a warmed cache must not lose to a cold one: {warm_rate:.3f} vs {lfu_rate:.3f}")
    check(warm_rate >= lru_rate + 0.02,
          f"the warm-up must buy a measurable margin: {warm_rate:.3f} vs {lru_rate:.3f}")
    check(all(rate <= optimal + 1e-9 for rate in (none_rate, lru_rate, lfu_rate, warm_rate)),
          f"no policy may beat the offline optimum {optimal:.4f}")
    check(optimal > lfu_rate, "the offline optimum must be a strict upper bound here")
    hottest = sorted(history, key=lambda feature: (-history[feature], feature))[:capacity]
    check(len(set(hottest)) == capacity, "the preload set must have the requested size")

    # 7. the cache's capacity invariant holds after every access, including at the edges
    probe = RowCache(4, "lru")
    for feature in [1, 2, 3, 4, 5, 6, 1, 2, 3, 7]:
        probe.access(feature)
        check(len(probe.entries) <= 4, "the cache must never exceed its capacity")
    check(len(probe.entries) == 4, "a full cache must stay full")
    distinct_rows = len(set(traffic))
    big_cache = RowCache(FEATURES * 2, "lru").run(traffic)
    check(abs(big_cache - (1 - distinct_rows / float(len(traffic)))) < 1e-12,
          f"a cache larger than the table must miss exactly once per distinct row, got "
          f"{big_cache:.4f}")
    check(RowCache(0, "lru").run(traffic) == 0.0, "a zero-capacity cache never hits")
    check(RowCache(4, "none").run([]) == 1.0, "an empty trace has nothing to miss")
    try:
        RowCache(4, "random")
        check(False, "an unknown eviction policy must be rejected")
    except AssertionError:
        pass

    # 8. determinism of the whole pipeline
    check(EmbeddingTable(FEATURES, shards=SHARDS).rows == table.rows,
          "the table fixture must rebuild identically")
    check(zipf_traffic(Rng(0x7A11), 6000, FEATURES) == traffic,
          "the traffic trace must be reproducible")
    check(frequency_history(Rng(0x4EAD), FEATURES) == history,
          "the frequency sketch must be reproducible")
    check(RowCache(capacity, "lfu").run(traffic) == lfu_rate,
          "the cache simulation must be reproducible")
    check(optimal_hit_rate(traffic, capacity) == optimal, "the offline optimum must be reproducible")

    print(f"table: {FEATURES} features in {SHARDS} shards {table.sizes()}, "
          f"{len(touched)} rows per batch, sparse == dense oracle over {6} steps")
    print(f"warm-up: first step {ramp_step:.4f} vs constant {flat_step:.4f}; "
          f"stable-peak loss {stable[-1]:.4f} vs {flat_stable[-1]:.4f}")
    print(f"cache: none={none_rate:.3f} lru={lru_rate:.3f} lfu={lfu_rate:.3f} "
          f"warmed={warm_rate:.3f} offline-optimal={optimal:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
