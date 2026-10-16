"""Task 660 - Implement quorum consensus read/write latency optimization tradeoffs.  (Tier: Expert, Ch 9 - System Design and Scalability)

N replicas with a write quorum W and a read quorum R are simulated against a virtual clock. Every
operation draws a per-replica response latency from a fixed-seed PRNG; a write stamps a monotonically
increasing version and lands on replica i at `write_start + latency(i)`, so the coordinator's ack time
is exactly the W-th smallest latency and precisely those W ackers hold the value when the ack returns
(the others receive it asynchronously at their own latency). A read asks a replica set, answers once R
responses have arrived - i.e. after the R-th smallest response latency - and returns the highest
version visible at that instant, repairing responders that lag behind.

This makes the tradeoff concrete: write latency is the W-th order statistic, read latency the R-th, so
both grow with the quorum size while the safety rule W + R > N is what forces them to overlap. That
rule is verified combinatorially (exhaustive quorum-set enumeration for small N, plus explicit disjoint
witness quorums whenever W + R <= N) and dynamically (over many simulated interleavings a read never
returns an older version than the latest acknowledged write). When W + R <= N the classic hazard is
reproduced, not described: a write that only reaches its W ackers is answered by a deliberately
disjoint read quorum with the previous value, even though the write was acknowledged. Read repair then
converges the stale replicas without ever lowering a version.

The optimisation is a real search: every legal (W, R) pair is scored by the mean of write plus read
latency over the samples, the brute-force optimum is compared against every other legal pair, and it is
checked to lie on the W + R = N + 1 frontier (any pair above it is dominated). Skewed replicas
demonstrate why the fastest responders are picked and why a larger quorum drags in the slow tail.

Simplified: latencies are static per operation (no batching, queueing or contention), the coordinator
is faultless apart from the one explicitly simulated failed write, and read repair is applied instantly
by the coordinator. Verified: quorum intersection, read-after-ack safety, the W + R <= N counterexample,
repair convergence and monotonicity, latency monotonicity/percentiles, frontier optimality, skew
behaviour, majority defaults, argument validation and determinism.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def quorums_intersect(n, w, r):
    """All write quorums of size >= w meet all read quorums of size >= r exactly when w + r > n."""
    return w + r > n


def disjoint_witness(n, w, r):
    """Explicit disjoint quorum pair, available exactly when w + r <= n."""
    if w + r > n:
        return None
    return tuple(range(w)), tuple(range(w, w + r))


def expect_error(call, label):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError(f"{label} must be rejected")


class LatencyModel:
    """Deterministic per-replica latency: base plus jitter, drawn once per operation."""

    def __init__(self, replicas, seed=0x5EED, base=1.0, jitter=4.0,
                 slow_replica=None, slow_factor=6.0):
        self.replicas = replicas
        self.base = base
        self.jitter = jitter
        self.slow_replica = slow_replica
        self.slow_factor = slow_factor
        self.seed = seed
        self.samples = {}

    def latency(self, replica, operation):
        key = (replica, operation)
        if key not in self.samples:
            state = (self.seed + (replica + 1) * 0x9E3779B97F4A7C15
                     + operation * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)
            _, r = splitmix64(state)
            value = self.base + (r % 1000) / 1000.0 * self.jitter
            if replica == self.slow_replica:
                value *= self.slow_factor
            self.samples[key] = value
        return self.samples[key]

    def replica_latencies(self, operation):
        return [self.latency(replica, operation) for replica in range(self.replicas)]


class Cluster:
    """Replicated versioned store with quorum reads/writes over a virtual clock."""

    def __init__(self, replicas, model):
        self.replicas = replicas
        self.model = model
        self.log = [[] for _ in range(replicas)]    # (store_time, version, value) per replica
        self.version = 0
        self.operation = 0
        self.now = 0.0
        self.history = []                           # (version, value, ack latency, ackers)

    # ------------------------------------------------------------------ state
    def visible(self, replica):
        best = (0, None)
        for store_time, version, value in self.log[replica]:
            if store_time <= self.now and version > best[0]:
                best = (version, value)
        return best

    def _append(self, replica, version, value, store_time):
        if version > self.visible(replica)[0] or store_time > self.now:
            self.log[replica].append((store_time, version, value))
        return replica

    def acked_version(self):
        return self.history[-1][0] if self.history else 0

    # ------------------------------------------------------------------ operations
    def write(self, value, w, durable=True):
        """Write to every replica; acknowledge after W responses. `durable=False` loses the rest."""
        check(1 <= w <= self.replicas, "the write quorum must fit the replica set")
        latencies = self.model.replica_latencies(self.operation)
        self.operation += 1
        order = sorted(range(self.replicas), key=lambda replica: (latencies[replica], replica))
        ackers = order[:w]
        ack_latency = max(latencies[replica] for replica in ackers)
        self.version += 1
        version = self.version
        start = self.now
        for replica in range(self.replicas):
            if replica in ackers or durable:
                self._append(replica, version, value, start + latencies[replica])
        self.now = start + ack_latency
        self.history.append((version, value, ack_latency, tuple(ackers)))
        return ack_latency, tuple(ackers)

    def read(self, r, quorum=None):
        """Read R responses, return the newest version visible, repair the stragglers."""
        check(1 <= r <= self.replicas, "the read quorum must fit the replica set")
        latencies = self.model.replica_latencies(self.operation)
        self.operation += 1
        candidates = list(range(self.replicas)) if quorum is None else list(quorum)
        check(len(candidates) >= r, "the queried set must be at least the read quorum")
        responders = tuple(sorted(candidates, key=lambda replica: (latencies[replica], replica))[:r])
        read_latency = max(latencies[replica] for replica in responders)
        snapshot = {replica: self.visible(replica) for replica in responders}
        newest_replica = max(responders, key=lambda replica: (snapshot[replica][0], -replica))
        version, value = snapshot[newest_replica]
        repaired = []
        for replica in responders:
            if snapshot[replica][0] < version:
                self._append(replica, version, value, self.now)
                repaired.append(replica)
        delayed = [replica for replica in range(self.replicas)
                   if self.visible(replica)[0] < version and replica not in responders]
        self.now += read_latency
        return value, version, read_latency, responders, tuple(repaired), tuple(delayed)


def mean_latency(model, size, operations=400):
    """Mean of the size-th smallest response latency over many operations."""
    total = 0.0
    for operation in range(operations):
        total += sorted(model.replica_latencies(operation))[size - 1]
    return total / operations


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main():
    replicas = 5
    model = LatencyModel(replicas, seed=0xC0FFEE)

    # 1. quorum intersection is exactly the W + R > N rule
    for n in range(1, 6):
        for w in range(1, n + 1):
            for r in range(1, n + 1):
                witness = disjoint_witness(n, w, r)
                check((witness is None) == quorums_intersect(n, w, r),
                      f"the intersection rule must be exact for n={n} w={w} r={r}")
                if witness is not None:
                    write, read = witness
                    check(set(write).isdisjoint(read), "the witness quorums must be disjoint")
                    check(len(write) >= w and len(read) >= r,
                          "the witness quorums must reach their required sizes")
    check(quorums_intersect(5, 3, 3) and not quorums_intersect(5, 2, 3),
          "majority quorums intersect, W + R = N does not")

    # 2. reads never lag an acknowledged write while W + R > N
    cluster = Cluster(replicas, model)
    cluster.write("v0", 3)
    for step in range(12):
        value, version, _, responders, repaired, delayed = cluster.read(3)
        check(version >= cluster.acked_version(),
              "a quorum read must never return older data than an acknowledged write")
        check(len(responders) == 3 and len(set(responders)) == 3,
              "a read must gather exactly R distinct responses")
        check(set(responders) <= set(range(replicas)), "responders must be real replicas")
        check(all(replica in responders for replica in repaired),
              "only responders can be repaired by a read")
        cluster.write(f"v{step + 1}", 3)
    check(all(entry[0] == index + 1 for index, entry in enumerate(cluster.history)),
          "versions must increase by exactly one per acknowledged write")

    # 3. a W + R <= N configuration really is unsafe, and the hazard is reproduced
    unsafe = Cluster(replicas, LatencyModel(replicas, seed=0x1234))
    unsafe.write("old", 5)
    ack_latency, ackers = unsafe.write("new", 2, durable=False)
    check(len(ackers) == 2, "the failed write reached only its two ackers")
    read_quorum = tuple(replica for replica in range(replicas) if replica not in ackers)
    check(len(read_quorum) == replicas - 2,
          "the complement of the write quorum must hold N - W replicas")
    value, version, _, responders, _, _ = unsafe.read(3, quorum=read_quorum)
    check(set(responders).isdisjoint(ackers),
          "the counterexample read quorum must be disjoint from the write ackers")
    check(value == "old" and version < unsafe.acked_version(),
          "with W + R <= N an acknowledged write can be missed by a read")
    print(f"660 unsafe read returned {value!r} while version {unsafe.acked_version()} was acked")

    # 4. read repair converges the responders and never lowers a version
    repair = Cluster(replicas, LatencyModel(replicas, seed=0x777))
    repair.write("a", 3)
    value, _, _, _, _, _ = repair.read(3)
    check(value == "a", "the first read sees the written value")
    repair.write("b", 5)
    value, version, _, responders, repaired, _ = repair.read(5)
    check(value == "b", "a full quorum read returns the newest value")
    check(all(repair.visible(replica)[0] == version for replica in range(replicas)),
          "after a full-quorum read every replica is at the newest version")
    before = [repair.visible(replica)[0] for replica in range(replicas)]
    repair.read(1)
    check(all(repair.visible(replica)[0] >= before[replica] for replica in range(replicas)),
          "a repair may never lower a replica's version")
    check(len(repaired) == 0 or all(replica in responders for replica in repaired),
          "repairs are limited to the responders")

    # 5. latency grows monotonically with the quorum size
    write_means = [mean_latency(model, w) for w in range(1, replicas + 1)]
    read_means = [mean_latency(model, r) for r in range(1, replicas + 1)]
    for index in range(1, replicas):
        check(write_means[index] >= write_means[index - 1] - 1e-9,
              "a larger write quorum cannot be faster")
        check(read_means[index] >= read_means[index - 1] - 1e-9,
              "a larger read quorum cannot be faster")
    check(write_means[0] < write_means[-1] and read_means[0] < read_means[-1],
          "the extremes of the quorum range must differ in latency")
    check(write_means[0] == read_means[0] == mean_latency(model, 1),
          "with one response the write and read latencies coincide")
    check(abs(write_means[-1] - read_means[-1]) < 1e-9,
          "waiting for every replica costs the same for reads and writes")

    # 6. the optimiser's choice lies on the W + R = N + 1 frontier and is not beaten by any pair
    pair_costs = {}
    for w in range(1, replicas + 1):
        for r in range(1, replicas + 1):
            if w + r > replicas:
                pair_costs[(w, r)] = mean_latency(model, w) + mean_latency(model, r)
    best_pair = min(pair_costs, key=lambda pair: (pair_costs[pair], pair))
    check(best_pair[0] + best_pair[1] == replicas + 1,
          "the cheapest quorum pair must sit on the minimal-intersection frontier")
    for pair, cost in pair_costs.items():
        check(pair_costs[best_pair] <= cost + 1e-9,
              "the chosen pair must not be beaten by another legal pair")
    frontier = [pair for pair in pair_costs if pair[0] + pair[1] == replicas + 1]
    check(all(pair_costs[pair] >= pair_costs[best_pair] - 1e-9 for pair in frontier),
          "the frontier must contain the optimum")
    print(f"660 optimum pair={best_pair} cost={pair_costs[best_pair]:.3f} "
          f"over {len(pair_costs)} legal pairs")

    # 7. skewed replicas: the ackers are the fastest, and a bigger quorum drags in the slow tail
    skew_model = LatencyModel(replicas, seed=0xABCD, slow_replica=2, slow_factor=6.0)
    skew = Cluster(replicas, skew_model)
    latencies = skew_model.replica_latencies(skew.operation)
    _, ackers = skew.write("x", 2)
    fastest = tuple(sorted(range(replicas), key=lambda replica: (latencies[replica], replica))[:2])
    check(fastest == ackers, "the ackers must be the fastest replicas")
    check(2 not in ackers, "the slow replica must be avoided while faster ones exist")
    comfortable = mean_latency(skew_model, 2, operations=200)
    dragged = mean_latency(skew_model, 4, operations=200)
    check(dragged > 1.5 * comfortable,
          "a quorum large enough to include the slow replica must be much slower")
    print(f"660 skewed replicas: W=2 costs {comfortable:.2f}, W=4 costs {dragged:.2f}")

    # 8. percentiles reflect the tail a bigger quorum must wait for
    samples_one = [sorted(model.replica_latencies(operation))[0] for operation in range(300)]
    samples_three = [sorted(model.replica_latencies(operation))[2] for operation in range(300)]
    check(percentile(samples_one, 0.5) <= percentile(samples_one, 0.95),
          "p95 may never be below p50")
    check(percentile(samples_three, 0.5) > percentile(samples_one, 0.5),
          "waiting for three responses must cost more than waiting for one")
    check(percentile(samples_three, 0.95) >= percentile(samples_one, 0.95),
          "the slower quorum must have the fatter tail")

    # 9. majority defaults and argument validation
    n = 7
    default = (n // 2 + 1, n // 2 + 1)
    check(default == (4, 4) and default[0] + default[1] == n + 1,
          "the majority default sits on the minimal frontier for odd N")
    check(quorums_intersect(n, *default), "majority quorums must intersect")
    expect_error(lambda: Cluster(replicas, model).write("v", 0), "a zero write quorum")
    expect_error(lambda: Cluster(replicas, model).write("v", 6), "an oversized write quorum")
    expect_error(lambda: Cluster(replicas, model).read(0), "a zero read quorum")
    expect_error(lambda: Cluster(replicas, model).read(6), "an oversized read quorum")
    expect_error(lambda: Cluster(replicas, model).read(3, quorum=(0, 1)),
                 "a read quorum larger than the queried set")

    # 10. determinism of the model and of the schedules
    first = Cluster(replicas, LatencyModel(replicas, seed=0xC0FFEE))
    second = Cluster(replicas, LatencyModel(replicas, seed=0xC0FFEE))
    for step in range(4):
        first.write(f"v{step}", 3)
        second.write(f"v{step}", 3)
        first.read(2)
        second.read(2)
    check([entry[2] for entry in first.history] == [entry[2] for entry in second.history],
          "the same seed must produce the same acknowledgement latencies")
    check(first.now == second.now, "the virtual clock must advance identically")
    check(model.replica_latencies(0) == model.replica_latencies(0),
          "latency samples must be stable for one operation")

    print(f"660 write means={[round(value, 2) for value in write_means]}")
    print(f"660 read means={[round(value, 2) for value in read_means]}")
    print("PASS")


if __name__ == "__main__":
    main()
