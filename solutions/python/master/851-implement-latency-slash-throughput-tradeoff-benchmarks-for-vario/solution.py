"""Task 851 - Implement latency-slash-throughput tradeoff benchmarks for various consistency levels.  (Tier: Master, Ch 9 — System Design and Scalability)

A quorum-replicated key-value store is benchmarked across consistency levels by a
deterministic discrete-event model.  There are no threads, sockets, sleeps or wall-clock
reads: virtual time is a float that only moves forward, every replica is a state machine
with its own FIFO queue, and the workload is a fixed pool of closed-loop clients whose
events are consumed in virtual-time order from an explicit heap.

What is simplified (a real cluster is replaced by an in-process event schedule):
  * `Replica` owns a one-way delay `one_way`, a service time `service` (spent applying one
    operation at the head of its queue) and the time `next_free` at which it is released.
    An operation dispatched at virtual time t is acknowledged at
    `max(t + one_way, next_free) + service + one_way`, so operations on a replica are
    serialised FIFO, can never overlap, and a request's latency is the *maximum*
    acknowledgement time over the replicas it was dispatched to.
  * `Level` is (name, write quorum W, read quorum R): ONE = (1,1), QUORUM = (N//2+1,
    N//2+1), ALL = (N,N), plus EVENTUAL = (1,1) whose writes are fire-and-forget (the client
    is acknowledged after a single hop, replication lands afterwards) - exactly the property
    that makes EVENTUAL unusable for read-your-writes.
  * A transaction is a write followed by a read; the read is dispatched only when the write
    has been acknowledged, so the two hops overlap across concurrent requests exactly like
    they would in a real cluster.  A coordinator dispatches to a preference-ordered subset
    of size k with a rotating start; writes carry monotone versions, and a read returns the
    freshest version among its R answers and repairs the laggards (read repair).
  * The load generator is a closed loop of W clients: every client reissues the instant its
    previous transaction completes, which is the canonical way to measure a saturation
    throughput.  The server throughput is therefore `W / mean_latency` (Little's law).

Invariants verified (safety, liveness and the cost/latency bounds):
  * quorum arithmetic, plus the intersection theorem brute-forced over every R-subset and
    W-subset of small rings: R + W > N  <=>  every read quorum meets every write quorum;
  * uncongested latency equals the analytic cost of both hops recomputed independently, and
    stays inside the envelope [2*(2*ow_min + service), 2*(2*ow_max + service)];
  * safety at scale: a read may never lag a write committed before it was dispatched
    (checked over every one of thousands of concurrent transactions).  R + W > N levels
    report zero violations while the W=R=1 and EVENTUAL controls demonstrably do violate it,
    so the check is not vacuous;
  * the trade-off is real and monotone: saturated throughput falls with the quorum size and
    never exceeds the analytic replica-capacity bound N/((W+R)*service), yet captures a
    measured fraction of it; latency rises in the opposite direction;
  * schedule well-formedness: no replica is ever double-booked, utilisation never exceeds
    1.0, the operation count is exactly `accepted * (W + R)`, and latency measured from the
    event trace satisfies Little's law against the independent W / mean_latency figure;
  * liveness: every client is served, queues settle after the last acknowledgement, and the
    whole benchmark is bit-for-bit reproducible from its seed.
"""

import heapq

MASK64 = (1 << 64) - 1
EPS = 1e-9


class SplitMix64:
    """Deterministic PRNG (Steele et al.): seeded replica skew without touching `random`."""

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


class Level:
    def __init__(self, name, write_quorum, read_quorum, async_replication=False):
        self.name = name
        self.w = write_quorum
        self.r = read_quorum
        self.async_replication = async_replication

    def __repr__(self):
        return "%s(W=%d,R=%d)" % (self.name, self.w, self.r)


def levels_for(n):
    """The canonical consistency levels of an N-replica ring."""
    majority = (n // 2) + 1
    return [
        Level("ONE", 1, 1),
        Level("QUORUM", majority, majority),
        Level("ALL", n, n),
        Level("EVENTUAL", 1, 1, async_replication=True),
    ]


class Replica:
    def __init__(self, index, one_way, service):
        self.index = index
        self.one_way = one_way
        self.service = service
        self.next_free = 0.0
        self.version = 0
        self.value = None
        self.ops = 0
        self.busy = []            # (start, end) intervals, for the no-overlap invariant

    def schedule(self, now, service=None):
        """Reserve the replica; return the time its acknowledgement reaches the client."""
        duration = self.service if service is None else service
        start = max(now + self.one_way, self.next_free)
        self.next_free = start + duration
        self.ops += 1
        if duration > 0.0:
            self.busy.append((start, start + duration))
        return self.next_free + self.one_way


class Cluster:
    """An N-replica ring with an explicit, deterministic in-memory network."""

    def __init__(self, n, seed=12345, base_one_way=1.0, service=0.2, skew=0.6):
        rng = SplitMix64(seed)
        self.n = n
        self.service = service
        self.replicas = [Replica(i, base_one_way * (1.0 + skew * rng.uniform(-1.0, 1.0)), service)
                         for i in range(n)]
        self.levels = {lv.name: lv for lv in levels_for(n)}
        self.sequence = 0
        self.version = 0
        self.committed_value = None
        self.committed_version = 0

    def preference_order(self, offset, count):
        """Rotating start spreads the load; the order is a pure function of `offset`."""
        return [self.replicas[(offset + j) % self.n] for j in range(count)]

    def write_targets(self, sequence, level):
        return self.preference_order(sequence % self.n, level.w)

    def read_targets(self, sequence, level):
        return self.preference_order((sequence * 3 + 1) % self.n, level.r)

    def write(self, now, value, level):
        """Dispatch a write to W replicas; returns (ack_time, targets)."""
        self.sequence += 1
        self.version += 1
        version = self.version
        targets = self.write_targets(self.sequence, level)
        acks = []
        for replica in targets:
            if level.async_replication:
                replica.schedule(now, service=0.0)      # fire-and-forget: absorbs the op
                acks.append(now + replica.one_way)
            else:
                acks.append(replica.schedule(now))
        for replica in targets:
            if replica.version < version:
                replica.version = version
                replica.value = value
        self.committed_version = version
        self.committed_value = value
        return max(acks), targets

    def read(self, now, level):
        """Dispatch a read to R replicas; returns (ack_time, version, value, targets)."""
        self.sequence += 1
        targets = self.read_targets(self.sequence, level)
        acks = [replica.schedule(now) for replica in targets]
        best = max(targets, key=lambda rep: (rep.version, -rep.index))
        for replica in targets:                          # read repair
            if replica.version < best.version:
                replica.version = best.version
                replica.value = best.value
        return max(acks), best.version, best.value, targets

    def transaction(self, now, value, level):
        """Sequential write-then-read; returns (ack_time, stale).  No other client runs."""
        wack, _ = self.write(now, value, level)
        rack, version, _val, _reps = self.read(wack, level)
        return rack, version < self.committed_version


def run_uncongested(cluster, level, rounds):
    """One client with a long idle gap between ops, so every request meets empty queues.

    A transaction is a write hop followed by a read hop; with no queueing its latency must
    equal the analytic cost of both hops, recomputed here from the replica delays alone.
    """
    now = max([rep.next_free for rep in cluster.replicas] + [0.0]) + 1000.0
    samples = []
    for _ in range(rounds):
        write_cost = max(2.0 * rep.one_way + cluster.service
                         for rep in cluster.write_targets(cluster.sequence + 1, level))
        read_cost = max(2.0 * rep.one_way + cluster.service
                        for rep in cluster.read_targets(cluster.sequence + 2, level))
        ack, _stale = cluster.transaction(now, "x", level)
        samples.append((ack - now, write_cost + read_cost))
        now = ack + 1000.0
    return samples


def run_closed_loop(cluster, level, workers, horizon, warmup):
    """The at-scale benchmark: `workers` clients, each reissuing as soon as it is served.

    Events ("issue" and "read") are consumed in virtual-time order, so concurrent
    transactions genuinely overlap.  Only transactions *issued* after `warmup` enter the
    steady-state window, which discards the cold start in which every queue is empty.
    Returns throughput, latency percentiles, and the safety counters.
    """
    counter = workers
    heap = [(0.0, w, "issue", w, 0.0) for w in range(workers)]
    heapq.heapify(heap)
    issued = 0
    reads = 0
    latencies = []
    stale = 0
    early_stale = 0
    while heap:
        now, _tie, kind, worker, start = heapq.heappop(heap)
        if now >= horizon:
            break
        if kind == "issue":
            issued += 1
            wack, _targets = cluster.write(now, ("v", issued), level)
            heapq.heappush(heap, (wack, counter, "read", worker, now))
            counter += 1
        else:
            reads += 1
            # a read may not lag any write that committed before the read was dispatched
            committed = cluster.committed_version
            rack, version, _val, _targets = cluster.read(now, level)
            if version < committed:
                if start >= warmup:
                    stale += 1
                else:
                    early_stale += 1
            if start >= warmup:
                latencies.append(rack - start)
            heapq.heappush(heap, (rack, counter, "issue", worker, rack))
            counter += 1
    latencies.sort()
    window = horizon - warmup
    measured = len(latencies)
    return {
        "issued": issued,
        "reads": reads,
        "measured": measured,
        "window": window,
        "throughput": measured / window if window > 0 else 0.0,
        "mean_latency": sum(latencies) / measured if measured else 0.0,
        "p99_latency": latencies[int(0.99 * (measured - 1))] if measured else 0.0,
        "max_latency": latencies[-1] if measured else 0.0,
        "stale_reads": stale,
        "early_stale": early_stale,
    }


def quorum_intersects(n, r, w):
    """Brute force: does every r-subset of an n-ring meet every w-subset?"""
    from itertools import combinations
    writes = [set(c) for c in combinations(range(n), w)]
    for readers in combinations(range(n), r):
        read_set = set(readers)
        for writer in writes:
            if not (read_set & writer):
                return False
    return True


def capacity_bound(n, level, service):
    """Replica service is the only finite resource: N/((W+R)*service) transactions per unit."""
    return n / ((level.w + level.r) * service)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. quorum arithmetic -------------------------------------------
    n = 5
    levels = {lv.name: lv for lv in levels_for(n)}
    check((levels["ONE"].w, levels["ONE"].r) == (1, 1), "ONE must be (1,1)")
    check(levels["QUORUM"].w == levels["QUORUM"].r == 3, "QUORUM of N=5 must be 3")
    check((levels["ALL"].w, levels["ALL"].r) == (5, 5), "ALL must be N")
    check(levels["EVENTUAL"].async_replication is True, "EVENTUAL must not wait for replicas")
    print("levels: %s" % [levels[k] for k in ("ONE", "QUORUM", "ALL", "EVENTUAL")])

    # ---- 2. quorum intersection theorem, brute forced ---------------------
    for size in range(1, 7):
        for w in range(1, size + 1):
            for r in range(1, size + 1):
                check(quorum_intersects(size, r, w) == (r + w > size),
                      "intersection theorem violated for N=%d R=%d W=%d" % (size, r, w))
    check(quorum_intersects(5, 1, 1) is False, "W=R=1 must not intersect")
    check(quorum_intersects(5, 3, 3) is True, "W=R=3 on N=5 must intersect")

    service = 0.2
    cluster = Cluster(n, seed=7, service=service)

    # ---- 3. uncongested latency == the analytic two-hop cost ------------
    for name in ("ONE", "QUORUM", "ALL"):
        for observed, expected in run_uncongested(cluster, levels[name], 30):
            check(abs(observed - expected) < 1e-9,
                  "%s: latency %.9f != analytic %.9f" % (name, observed, expected))
    means = {}
    ow = [rep.one_way for rep in cluster.replicas]
    floor = 2.0 * (2.0 * min(ow) + service)
    ceiling = 2.0 * (2.0 * max(ow) + service)
    for name in ("ONE", "QUORUM", "ALL"):
        samples = run_uncongested(cluster, levels[name], 60)
        means[name] = sum(o for o, _ in samples) / len(samples)
        check(means[name] >= floor - EPS, "%s below the analytic latency floor" % name)
        check(means[name] <= ceiling + EPS, "%s above the analytic latency ceiling" % name)
    print("uncongested mean latency: ONE=%.3f QUORUM=%.3f ALL=%.3f (envelope %.3f..%.3f)"
          % (means["ONE"], means["QUORUM"], means["ALL"], floor, ceiling))
    check(means["ONE"] <= means["QUORUM"] <= means["ALL"],
          "latency must be monotone in the acknowledgement count")
    check(means["ONE"] < means["ALL"], "ALL must be strictly slower than ONE")

    # ---- 4. safety: no stale read when R + W > N (sequential client) ----
    stale_report = {}
    for name in ("ONE", "QUORUM", "ALL"):
        fresh = Cluster(n, seed=99, service=service)
        stale = 0
        for i in range(300):
            _ack, is_stale = fresh.transaction(float(i), i, levels[name])
            stale += 1 if is_stale else 0
        stale_report[name] = stale
        if levels[name].r + levels[name].w > n:
            check(stale == 0, "%s must never serve a stale read (R+W>N)" % name)
    print("sequential write-then-read stale reads: %s" % stale_report)
    check(stale_report["ONE"] > 0, "W=R=1 must be able to serve stale reads (control)")

    eventual = Cluster(n, seed=99, service=service)
    stale = 0
    for i in range(300):
        _ack, is_stale = eventual.transaction(float(i), i, levels["EVENTUAL"])
        stale += 1 if is_stale else 0
    print("EVENTUAL control stale reads: %d of 300" % stale)
    check(stale > 0, "the EVENTUAL negative control must observe stale reads")

    # ---- 5. the saturated latency/throughput trade-off ------------------
    horizon = 1200.0
    warmup = 600.0
    workers = 80
    runs = {}
    for name in ("ONE", "QUORUM", "ALL"):
        runs[name] = run_closed_loop(Cluster(n, seed=11, service=service),
                                     levels[name], workers, horizon, warmup)
        m = runs[name]
        bound = capacity_bound(n, levels[name], service)
        check(m["throughput"] <= bound + EPS,
              "%s: measured throughput %.3f exceeds the replica-capacity bound %.3f"
              % (name, m["throughput"], bound))
        # the schedule is not perfectly efficient (network hops idle a replica), but a
        # well-formed closed loop captures the large majority of the analytic bound
        check(m["throughput"] >= 0.85 * bound,
              "%s: measured throughput %.3f captures under 85%% of the bound %.3f"
              % (name, m["throughput"], bound))
        check(m["measured"] > 200, "%s: too few samples for a stable benchmark" % name)
        print("saturated %-7s throughput=%.3f (bound %.3f, %.0f%%) latency=%.3f p99=%.3f"
              % (name, m["throughput"], bound, 100 * m["throughput"] / bound,
                 m["mean_latency"], m["p99_latency"]))

    check(runs["ONE"]["throughput"] > runs["QUORUM"]["throughput"] > runs["ALL"]["throughput"],
          "saturated throughput must fall as the quorum grows")
    check(runs["ONE"]["mean_latency"] < runs["QUORUM"]["mean_latency"] < runs["ALL"]["mean_latency"],
          "saturated latency must rise as the quorum grows")
    check(runs["QUORUM"]["mean_latency"] > means["ONE"],
          "saturated QUORUM latency must exceed its uncongested latency (queueing happened)")

    # Little's law ties the measured throughput to the independently measured latency
    for name, m in runs.items():
        expected = workers / m["mean_latency"]
        rel = abs(m["throughput"] - expected) / expected
        check(rel < 0.05, "%s: Little's law off by %.2f%%" % (name, 100 * rel))
        print("Little's law %-7s throughput=%.3f workers/latency=%.3f error=%.2f%%"
              % (name, m["throughput"], expected, 100 * rel))

    # ---- 6. safety at scale ---------------------------------------------
    for name in ("QUORUM", "ALL"):
        m = runs[name]
        check(m["stale_reads"] == 0,
              "%s: %d of %d concurrent reads lagged a committed write"
              % (name, m["stale_reads"], m["measured"]))
    concurrent = run_closed_loop(Cluster(n, seed=11, service=service), levels["ONE"],
                                 workers, horizon, warmup)
    print("concurrent ONE stale reads: %d of %d" % (concurrent["stale_reads"], concurrent["measured"]))
    check(concurrent["stale_reads"] > 0,
          "the concurrent W=R=1 control must observe stale reads at scale")

    # ---- 7. schedule well-formedness (replica invariants) ---------------
    probe = Cluster(n, seed=11, service=service)
    metrics = run_closed_loop(probe, levels["QUORUM"], workers, horizon, warmup)
    for replica in probe.replicas:
        intervals = replica.busy
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            check(e1 <= s2 + EPS,
                  "replica %d double-booked: [%.3f,%.3f] overlaps [%.3f,%.3f]"
                  % (replica.index, s1, e1, s2, e2))
        check(intervals == sorted(intervals), "replica %d served ops out of order" % replica.index)
        busy = sum(max(0.0, min(e, horizon) - max(s, 0.0)) for s, e in intervals)
        check(busy <= horizon + EPS,
              "replica %d utilisation %.3f exceeds 1.0 inside the benchmark window"
              % (replica.index, busy / horizon))
    level = levels["QUORUM"]
    total_ops = sum(rep.ops for rep in probe.replicas)
    check(total_ops == metrics["issued"] * level.w + metrics["reads"] * level.r,
          "operation count %d != issued*W + reads*R = %d"
          % (total_ops, metrics["issued"] * level.w + metrics["reads"] * level.r))
    check(metrics["reads"] <= metrics["issued"], "a read was dispatched before its write")
    check(metrics["issued"] - metrics["reads"] <= workers,
          "more than one read per client was left in flight at the horizon")
    check(metrics["issued"] >= metrics["measured"], "the window cannot hold more than was issued")

    # ---- 8. liveness: every client is served and the queues settle ------
    final = Cluster(n, seed=11, service=service)
    metrics = run_closed_loop(final, levels["ALL"], workers, horizon, warmup)
    settles = max(rep.next_free for rep in final.replicas)
    check(settles <= horizon + 60 * (2 * service + 3.0 * max(rep.one_way for rep in final.replicas)),
          "replica queues never settled after the last acknowledgement (%.3f)" % settles)
    check(metrics["stale_reads"] == 0, "ALL must never serve a stale read under load")
    check(metrics["max_latency"] >= metrics["p99_latency"] >= metrics["mean_latency"] > 0,
          "latency percentiles are inconsistent")

    # ---- 9. edge cases ---------------------------------------------------
    single_levels = levels_for(1)
    check([(lv.w, lv.r) for lv in single_levels[:3]] == [(1, 1), (1, 1), (1, 1)],
          "a single replica collapses every quorum to 1")
    check(quorum_intersects(1, 1, 1) is True, "N=1 quorums must intersect")
    single = run_closed_loop(Cluster(1, seed=3, service=service), single_levels[0], 8, 200.0, 100.0)
    check(single["measured"] > 10 and single["stale_reads"] == 0,
          "a single-replica cluster must be trivially safe")
    idle = run_closed_loop(Cluster(3, seed=1, service=service), levels_for(3)[0], 0, 100.0, 0.0)
    check(idle["issued"] == 0 and idle["mean_latency"] == 0.0,
          "no clients must produce an empty, non-crashing run")
    instant = run_closed_loop(Cluster(3, seed=1, service=service), levels_for(3)[0], 4, 0.0, 0.0)
    check(instant["issued"] == 0 and instant["throughput"] == 0.0,
          "a zero horizon must produce no throughput and no crash")
    check(levels_for(3)[0].async_replication is False, "a synchronous level must not be async")

    # ---- 10. reproducibility --------------------------------------------
    a = run_closed_loop(Cluster(5, seed=4242, service=service), levels["QUORUM"], 40, 300.0, 100.0)
    b = run_closed_loop(Cluster(5, seed=4242, service=service), levels["QUORUM"], 40, 300.0, 100.0)
    check(a == b, "the benchmark must be bit-for-bit reproducible from its seed")
    c = run_closed_loop(Cluster(5, seed=4243, service=service), levels["QUORUM"], 40, 300.0, 100.0)
    check(abs(c["mean_latency"] - a["mean_latency"]) > 0.0,
          "changing the seed must actually change the replica delays")

    print("PASS")


if __name__ == "__main__":
    main()
