"""Task 891 - Implement benchmarking harness for large distributed systems with failure injection.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

The harness is a deterministic discrete-event simulator: virtual time only, no threads, no
sockets, no sleeping, no wall clock.  Every state change is an event in a priority queue
ordered by (time, sequence number), so a run is a pure function of the configuration and the
seed, and the simulated clock is the quantity all latency and throughput numbers refer to.

The modelled system is a replicated key-value cluster.  A client sends requests whose arrival
times come from the seeded generator; each request is routed to one of the `replication`
replicas of its key, the target node serialises work with a FIFO single-server queue, and a
request that is refused, silently dropped or takes too long is retried against the next replica
up to `max_attempts` times.  Failure injection is itself scheduled on the same virtual clock:
node crashes and restarts, per-node slowdown factors, and a network partition that splits the
cluster into two groups (messages crossing it are silently dropped).  Healing is just another
scheduled fault, so "recovery" is measured rather than assumed.

Metrics are recorded, not estimated: success and failure counts by cause, retries, per-node
loads, and exact latency percentiles over the recorded samples.  The accounting is checked
against an independent quantity - the area under the in-flight curve, integrated exactly
between event times, must equal the sum of every request's latency - and the queueing numbers
are checked against closed forms (a single request takes exactly one service time, two
simultaneous requests to one node take one and two service times, n simultaneous requests to
one node end at n service times).

Verified: events are processed in nondecreasing time, the analytic latency of an unloaded
request equals the configured service time, FIFO queueing matches the closed form, the
percentile helper matches hand-computed nearest-rank values and a property sweep, the
in-flight area identity holds exactly, an RF=1 cluster loses exactly the requests whose key
maps to the crashed node while RF=3 completes all of them through retries, a slow node inflates
latencies by exactly its factor, a partition blocks every cross-group request and healing
restores a 100% success rate, a total outage injects zero successes while a no-op fault injects
zero failures, malformed configurations are rejected, and a 2000 request run is reproducible
from its seed.
"""
import heapq
import math

MASK64 = (1 << 64) - 1

ARRIVAL = 0
COMPLETION = 1
TIMEOUT = 2
FAULT = 3

REFUSED = "refused"
DROPPED = "dropped"
EXHAUSTED = "exhausted"


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

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def permutation_pick(self, span):
        """Uniform integer in [-span, span] used for the jittered arrival process."""
        return self.below(2 * span + 1) - span


class ConfigError(Exception):
    """Raised for a configuration the harness cannot honour."""


def percentile(samples, percent):
    """Exact nearest-rank percentile over the recorded samples."""
    if not samples:
        raise ValueError("no latency samples were recorded")
    if percent <= 0 or percent > 100:
        raise ValueError("the percentile must lie in (0, 100]")
    ordered = sorted(samples)
    rank = math.ceil(percent / 100.0 * len(ordered))
    return ordered[max(1, rank) - 1]


def percentile_by_index(samples, percent):
    """Second implementation of the same definition, used to cross-check `percentile`."""
    ordered = sorted(samples)
    rank = -(-percent * len(ordered) // 100)
    return ordered[rank - 1]


class Simulator:
    """Virtual-time event queue; ties break on the insertion sequence for determinism."""

    def __init__(self):
        self.events = []
        self.sequence = 0
        self.clock = 0
        self.processed = 0
        self.last_processed = 0

    def schedule(self, delay, kind, payload):
        if delay < 0:
            raise ConfigError("an event cannot be scheduled in the past")
        self.sequence += 1
        heapq.heappush(self.events, (self.clock + delay, self.sequence, kind, payload))

    def schedule_at(self, time, kind, payload):
        if time < self.clock:
            raise ConfigError("an event cannot be scheduled in the past")
        self.sequence += 1
        heapq.heappush(self.events, (time, self.sequence, kind, payload))

    def run(self, on_event):
        while self.events:
            time, _, kind, payload = heapq.heappop(self.events)
            if time < self.last_processed:
                raise AssertionError("the event queue produced time %d after %d"
                                     % (time, self.last_processed))
            self.last_processed = time
            self.clock = time
            self.processed += 1
            on_event(kind, payload, time)


class Cluster:
    """Replicated nodes with crash, slowdown and partition state."""

    def __init__(self, node_count, service_time, replication, timeout, max_attempts=3):
        if node_count <= 0:
            raise ConfigError("a cluster needs at least one node")
        if replication <= 0 or replication > node_count:
            raise ConfigError("replication must lie in [1, node_count]")
        if service_time <= 0:
            raise ConfigError("the service time must be positive")
        if timeout <= 0:
            raise ConfigError("the timeout must be positive")
        if max_attempts <= 0:
            raise ConfigError("max_attempts must be positive")
        self.node_count = node_count
        self.service_time = service_time
        self.replication = replication
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.busy_until = [0] * node_count
        self.processed = [0] * node_count
        self.crashed = set()
        self.slowdown = {}
        self.partition = None
        self.refused = 0
        self.crossed = 0

    def replica_plan(self, key):
        base = (key * 2654435761) % 4294967296
        return [(base + step) % self.node_count for step in range(self.replication)]

    def service_of(self, node):
        return self.service_time * self.slowdown.get(node, 1)

    def in_same_group(self, node):
        """The client lives with node 0, so a message to another group is dropped."""
        if self.partition is None:
            return True
        first, _ = self.partition
        return (0 in first) == (node in first)


class Harness:
    """Runs a workload against a cluster and records metrics."""

    def __init__(self, cluster, seed, request_count, mean_interval, jitter=2, keyspace=32,
                 fault_probability=0, faults=()):
        self.cluster = cluster
        self.rng = Rng(seed)
        self.request_count = request_count
        self.mean_interval = mean_interval
        self.jitter = jitter
        self.keyspace = keyspace
        self.fault_probability = fault_probability
        self.faults = list(faults)
        self.simulator = Simulator()
        self.requests = {}
        self.in_flight = 0
        self.area = 0
        self.previous_time = 0
        self.latencies = []
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.failure_causes = {}
        self.latency_by_node = {}
        self.next_request_id = 0
        self.frozen = False
        self.arrival_time = 0

    # -------------------------------------------------------------- workload
    def _next_arrival(self):
        step = self.mean_interval + self.rng.permutation_pick(self.jitter)
        return max(0, step)

    def build_workload(self):
        time = 0
        for _ in range(self.request_count):
            time += self._next_arrival()
            self.simulator.schedule_at(time, ARRIVAL, self.rng.below(self.keyspace))
        for when, action, argument in self.faults:
            self.simulator.schedule_at(when, FAULT, (action, argument))
        self.arrival_time = time
        return time

    # ----------------------------------------------------------------- main
    def run(self):
        self.build_workload()
        self.simulator.run(self._handle)
        return self.metrics()

    def _handle(self, kind, payload, time):
        self.area += self.in_flight * (time - self.previous_time)
        self.previous_time = time
        if kind == ARRIVAL:
            self._arrive(payload, time)
        elif kind == COMPLETION:
            self._complete(payload, time)
        elif kind == TIMEOUT:
            self._timeout(payload, time)
        else:
            self._fault(payload, time)

    def _arrive(self, key, time):
        request_id = self.next_request_id
        self.next_request_id += 1
        request = {"id": request_id, "key": key, "start": time, "attempt": 0, "done": False}
        self.requests[request_id] = request
        self.in_flight += 1
        self._dispatch(request, time)

    def _dispatch(self, request, time):
        cluster = self.cluster
        node = cluster.replica_plan(request["key"])[request["attempt"] % cluster.replication]
        if node in cluster.crashed:
            cluster.refused += 1
            self._retry(request, time, REFUSED)
            return
        if not cluster.in_same_group(node):
            cluster.crossed += 1
            self.simulator.schedule(cluster.timeout, TIMEOUT, request["id"])
            return
        if self.fault_probability and self.rng.below(100) < self.fault_probability:
            cluster.crossed += 1
            self.simulator.schedule(cluster.timeout, TIMEOUT, request["id"])
            return
        service = cluster.service_of(node)
        start = max(cluster.busy_until[node], time)
        cluster.busy_until[node] = start + service
        cluster.processed[node] += 1
        self.latency_by_node.setdefault(node, 0)
        self.simulator.schedule_at(cluster.busy_until[node], COMPLETION,
                                   (request["id"], node))
        self.simulator.schedule(cluster.timeout, TIMEOUT, request["id"])

    def _complete(self, payload, time):
        request_id, node = payload
        request = self.requests[request_id]
        if request["done"]:
            return
        self._finish(request, time, f"node{node}", success=True)

    def _timeout(self, payload, time):
        request = self.requests[payload]
        if request["done"]:
            return
        self._retry(request, time, DROPPED)

    def _retry(self, request, time, cause):
        request["attempt"] += 1
        if request["attempt"] >= self.cluster.max_attempts:
            self._finish(request, time, cause, success=False)
            return
        self.retries += 1
        self._dispatch(request, time)

    def _finish(self, request, time, label, success):
        request["done"] = True
        latency = time - request["start"]
        self.latencies.append(latency)
        self.in_flight -= 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
            self.failure_causes[label] = self.failure_causes.get(label, 0) + 1

    def _fault(self, payload, time):
        action, argument = payload
        cluster = self.cluster
        if action == "crash":
            cluster.crashed.add(argument)
        elif action == "restart":
            cluster.crashed.discard(argument)
        elif action == "slow":
            node, factor = argument
            cluster.slowdown[node] = factor
        elif action == "normal":
            cluster.slowdown.pop(argument, None)
        elif action == "partition":
            cluster.partition = (frozenset(argument[0]), frozenset(argument[1]))
        elif action == "heal":
            cluster.partition = None
        elif action == "freeze":
            self.frozen = True
        else:
            raise ConfigError("unknown fault %r" % (action,))

    # -------------------------------------------------------------- metrics
    def metrics(self):
        duration = self.simulator.clock - 0
        return {
            "arrivals": self.next_request_id,
            "arrival_window": self.arrival_time,
            "duration": duration,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "causes": dict(self.failure_causes),
            "latencies": list(self.latencies),
            "p50": percentile(self.latencies, 50) if self.latencies else None,
            "p90": percentile(self.latencies, 90) if self.latencies else None,
            "p99": percentile(self.latencies, 99) if self.latencies else None,
            "max": max(self.latencies) if self.latencies else None,
            "mean": (sum(self.latencies) / len(self.latencies)) if self.latencies else None,
            "throughput": self.successes / duration if duration else 0.0,
            "per_node": list(self.cluster.processed),
            "refused": self.cluster.refused,
            "crossed": self.cluster.crossed,
            "events": self.simulator.processed,
        }


# ----------------------------------------------------------------------- main
def run_harness(cluster, seed, request_count, mean_interval, **kwargs):
    return Harness(cluster, seed, request_count, mean_interval, **kwargs).run()


def main():
    # --- 1. percentile helper -----------------------------------------------
    samples = list(range(1, 101))
    check(percentile(samples, 50) == 50 and percentile(samples, 99) == 99
          and percentile(samples, 100) == 100, "nearest-rank percentiles must match 1..100")
    check(percentile([7], 1) == 7 and percentile([7], 99) == 7,
          "a single sample is its own percentile")
    check(percentile([5, 1, 9, 3, 7], 20) == 1 and percentile([5, 1, 9, 3, 7], 50) == 5
          and percentile([5, 1, 9, 3, 7], 90) == 9,
          "nearest-rank must pick the first sample at or above the requested rank")
    rng = Rng(891001)
    for _ in range(200):
        random_samples = [rng.below(1000) + 1 for _ in range(1 + rng.below(60))]
        for percent in (1, 25, 50, 75, 90, 99, 100):
            check(percentile(random_samples, percent)
                  == percentile_by_index(random_samples, percent),
                  "the two percentile implementations must agree")
        check(percentile(random_samples, 50) <= percentile(random_samples, 90)
              <= percentile(random_samples, 99) <= percentile(random_samples, 100),
              "percentiles must be monotone in the percentile level")
    raised = False
    try:
        percentile([], 50)
    except ValueError:
        raised = True
    check(raised, "an empty sample set has no percentile")

    # --- 2. one unloaded request costs exactly one service time ------------
    cluster = Cluster(node_count=3, service_time=7, replication=1, timeout=100)
    harness = Harness(cluster, 891100, request_count=1, mean_interval=10)
    metrics = harness.run()
    check(metrics["latencies"] == [7],
          "a single request must take exactly one service time, got %r" % metrics["latencies"])
    check(metrics["successes"] == 1 and metrics["failures"] == 0,
          "the single request must succeed")

    # --- 3. FIFO serialisation matches the closed form ----------------------
    cluster = Cluster(node_count=1, service_time=5, replication=1, timeout=1000)
    harness = Harness(cluster, 891101, request_count=6, mean_interval=0, jitter=0)
    metrics = harness.run()
    check(sorted(metrics["latencies"]) == [5, 10, 15, 20, 25, 30],
          "six simultaneous requests to one node must finish at n * service, got %r"
          % sorted(metrics["latencies"]))
    check(metrics["max"] == 30, "the last request must wait for five predecessors")

    # --- 4. the in-flight area identity ------------------------------------
    cluster = Cluster(node_count=4, service_time=4, replication=2, timeout=60)
    area_harness = Harness(cluster, 891200, request_count=120, mean_interval=2, jitter=1)
    area_metrics = area_harness.run()
    check(area_harness.in_flight == 0, "every request must have terminated by the end of the run")
    check(area_harness.area == sum(area_metrics["latencies"]),
          "the in-flight area %d must equal the sum of latencies %d"
          % (area_harness.area, sum(area_metrics["latencies"])))
    check(area_metrics["successes"] + area_metrics["failures"] == area_metrics["arrivals"],
          "every arrival must terminate")

    # --- 5. determinism -----------------------------------------------------
    def steady(seed):
        cluster = Cluster(node_count=4, service_time=4, replication=2, timeout=60)
        return Harness(cluster, seed, request_count=150, mean_interval=3, jitter=2).run()

    first = steady(891300)
    second = steady(891300)
    check(first == second, "an identical seed must reproduce every metric")
    check(first["latencies"] == second["latencies"], "the latency sequence must be identical")

    # --- 6. RF=1 loses exactly the requests aimed at the crashed node -------
    cluster = Cluster(node_count=3, service_time=3, replication=1, timeout=40, max_attempts=1)
    harness = Harness(cluster, 891400, request_count=90, mean_interval=4, jitter=1)
    harness.build_workload()
    crash_time = harness.arrival_time // 2
    harness.simulator.schedule_at(crash_time, FAULT, ("crash", 1))
    harness.simulator.run(harness._handle)
    crashed_metrics = harness.metrics()
    check(crashed_metrics["successes"] + crashed_metrics["failures"] == crashed_metrics["arrivals"],
          "the faulted run must still terminate every request")
    expected_failures = sum(
        1 for request in harness.requests.values()
        if request["start"] >= crash_time and cluster.replica_plan(request["key"])[0] == 1)
    check(expected_failures > 0, "the crash window must cover requests to the crashed node")
    check(crashed_metrics["failures"] == expected_failures,
          "exactly the requests routed to the crashed node must fail: %d vs %d"
          % (crashed_metrics["failures"], expected_failures))
    check(crashed_metrics["causes"] == {REFUSED: expected_failures},
          "every failure must be a refusal, got %r" % (crashed_metrics["causes"],))
    check(crashed_metrics["successes"] == crashed_metrics["arrivals"] - expected_failures,
          "the healthy keys must be unaffected by the crash")

    # --- 7. RF=3 survives the same crash through retries --------------------
    cluster = Cluster(node_count=3, service_time=3, replication=3, timeout=40, max_attempts=3)
    harness = Harness(cluster, 891400, request_count=90, mean_interval=4, jitter=1)
    harness.build_workload()
    window = harness.arrival_time
    harness.simulator.schedule_at(window // 3, FAULT, ("crash", 1))
    harness.simulator.schedule_at(window + 50, FAULT, ("restart", 1))
    harness.simulator.run(harness._handle)
    replicated = harness.metrics()
    check(replicated["failures"] == 0,
          "replication 3 must ride out a single crashed node, got %r" % (replicated["causes"],))
    check(replicated["retries"] > 0,
          "the surviving requests must have been retried against another replica")
    check(replicated["successes"] == replicated["arrivals"], "every request must succeed")

    # --- 8. a slow node inflates latencies by exactly its factor ------------
    factor = 4
    cluster = Cluster(node_count=1, service_time=6, replication=1, timeout=500)
    harness = Harness(cluster, 891500, request_count=1, mean_interval=10)
    harness.build_workload()
    harness.simulator.schedule_at(0, FAULT, ("slow", (0, factor)))
    harness.simulator.run(harness._handle)
    slow = harness.metrics()
    check(slow["latencies"] == [6 * factor],
          "a node slowed by %d must serve in %d units, got %r"
          % (factor, 6 * factor, slow["latencies"]))
    baseline = run_harness(Cluster(1, 6, 1, 500), 891501, 1, 10)
    check(slow["latencies"][0] > baseline["latencies"][0],
          "the injected slowdown must be visible in the latency")

    # --- 9. partition blocks cross-group traffic and healing restores it ----
    def partitioned_run(heal):
        cluster = Cluster(node_count=4, service_time=3, replication=1, timeout=25,
                          max_attempts=1)
        harness = Harness(cluster, 891600, request_count=80, mean_interval=3, jitter=1)
        harness.build_workload()
        window = harness.arrival_time
        harness.simulator.schedule_at(0, FAULT, ("partition", ([0, 1], [2, 3])))
        if heal is not None:
            harness.simulator.schedule_at(heal, FAULT, ("heal", None))
        harness.simulator.run(harness._handle)
        return harness.metrics()

    split = partitioned_run(None)
    check(split["failures"] > 0, "a permanent partition must produce failures")
    check(split["crossed"] > 0, "cross-group messages must be recorded as dropped")
    check(split["causes"].get(DROPPED) == split["failures"],
          "partition failures must be timeouts, got %r" % (split["causes"],))
    healed = partitioned_run(60)
    check(healed["successes"] > split["successes"],
          "healing the partition must recover requests: %d vs %d"
          % (healed["successes"], split["successes"]))
    fully_healed = partitioned_run(1)
    check(fully_healed["failures"] == 0,
          "healing at the very start must leave no failures at all")

    # --- 10. total outage versus no outage ---------------------------------
    outage = run_harness(Cluster(3, 3, 1, 20, max_attempts=1), 891700, 30, 3,
                         fault_probability=100)
    check(outage["successes"] == 0 and outage["failures"] == 30,
          "a 100% drop rate must inject exactly zero successes")
    check(outage["causes"].get(DROPPED) == 30, "every outage failure must be a drop")
    healthy = run_harness(Cluster(3, 3, 1, 20, max_attempts=1), 891700, 30, 3,
                          fault_probability=0)
    check(healthy["failures"] == 0 and healthy["successes"] == 30,
          "a zero drop rate must inject no failures at all")

    # --- 11. timeouts take exactly the configured timeout ------------------
    cluster = Cluster(node_count=1, service_time=1, replication=1, timeout=50, max_attempts=1)
    harness = Harness(cluster, 891800, request_count=1, mean_interval=10,
                      fault_probability=100)
    metrics = harness.run()
    check(metrics["latencies"] == [50],
          "a dropped request must fail at exactly the timeout, got %r" % metrics["latencies"])

    # --- 12. invalid configurations are rejected ---------------------------
    for bad in (lambda: Cluster(0, 5, 1, 10), lambda: Cluster(3, 5, 4, 10),
                lambda: Cluster(3, 0, 1, 10), lambda: Cluster(3, 5, 1, 0),
                lambda: Cluster(3, 5, 1, 10, max_attempts=0)):
        raised = False
        try:
            bad()
        except ConfigError:
            raised = True
        check(raised, "an invalid cluster configuration must be rejected")
    raised = False
    try:
        Harness(Cluster(2, 5, 1, 10), 891900, 5, 3, faults=[(10, "meltdown", None)]).run()
    except ConfigError:
        raised = True
    check(raised, "an unknown fault action must be rejected")

    # --- 13. scale ----------------------------------------------------------
    scale_cluster = Cluster(node_count=8, service_time=3, replication=3, timeout=40,
                            max_attempts=3)
    scale_faults = [(0, "crash", 2), (500, "restart", 2), (1500, "slow", (5, 3)),
                    (2000, "normal", 5), (2500, "partition", ([0, 1, 2, 3], [4, 5, 6, 7])),
                    (3000, "heal", None)]
    scale = run_harness(scale_cluster, 892000, 2000, 2, jitter=1, faults=scale_faults)
    check(scale["arrivals"] == 2000, "the scaled workload must generate every arrival")
    check(scale["successes"] + scale["failures"] == 2000, "every scaled request must terminate")
    check(scale["failures"] > 0, "the injected partition must cost some requests")
    check(scale["failures"] < scale["arrivals"] // 4,
          "losing a quarter of the fleet must not cost a quarter of the requests: %d"
          % scale["failures"])
    check(scale["causes"] == {DROPPED: scale["failures"]},
          "the partition must be the only source of failures, got %r" % (scale["causes"],))
    check(scale["retries"] > 0, "the scaled run must show retries")
    check(sum(scale["per_node"]) + scale["refused"] + scale["crossed"]
          == scale["arrivals"] + scale["retries"],
          "every dispatch must be exactly one accepted, refused or crossed message")
    check(scale["events"] > 4000, "the scaled run must process a substantial event stream")
    check(scale["p50"] <= scale["p90"] <= scale["p99"] <= scale["max"],
          "the percentile ordering must hold at scale")
    check(abs(scale["throughput"] * scale["duration"] - scale["successes"]) < 1e-6,
          "throughput must be successes per unit of simulated time")

    # --- 14. reproducibility of the full harness ---------------------------
    again = run_harness(Cluster(node_count=8, service_time=3, replication=3, timeout=40,
                                max_attempts=3), 892000, 2000, 2, jitter=1,
                        faults=scale_faults)
    check(again == scale, "the scaled run must be bit-for-bit reproducible")

    print("accounting: in-flight area %d == sum of latencies %d over %d requests"
          % (area_harness.area, sum(area_metrics["latencies"]), area_metrics["arrivals"]))
    print("steady state: p50 %d, p90 %d, p99 %d, max %d, %.3f requests per unit of time"
          % (first["p50"], first["p90"], first["p99"], first["max"], first["throughput"]))
    print("fault isolation: replication 1 -> %d failures, replication 3 -> %d failures"
          % (crashed_metrics["failures"], replicated["failures"]))
    print("partition: %d dropped while split, %d recovered after healing"
          % (split["crossed"], healed["successes"] - split["successes"]))
    print("scale: %d arrivals, %d events, %d retries, %d successes, p99 %d"
          % (scale["arrivals"], scale["events"], scale["retries"], scale["successes"],
             scale["p99"]))
    print("PASS")


if __name__ == "__main__":
    main()
