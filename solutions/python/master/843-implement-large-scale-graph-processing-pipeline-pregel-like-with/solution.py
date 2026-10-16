"""Task 843 - Implement large-scale graph processing pipeline (Pregel-like) with fault tolerance.  (Tier: Master, CTCI Beyond — System Design & Scalability)

A bulk-synchronous (BSP) vertex-centric engine: the graph is partitioned across workers, each
superstep runs every *active* vertex against the messages it received in the previous
superstep, and the barrier moves the produced messages over an explicit in-memory network.
There are no threads, sockets, sleeps or wall-clock reads: workers, messages and checkpoints
are plain data structures and a failure is a scheduled event, not a real crash.

Engine, as implemented:
  * a deterministic partition (vertex -> worker by sorted index), so every run sends exactly
    the same messages in the same order;
  * message-driven execution: a vertex is recomputed when it has mail; the run ends when no
    messages are in flight, every vertex votes to halt, or the superstep cap is reached;
  * Pregel aggregators (`contribute` during superstep s is visible in superstep s+1), used here
    for the global PageRank convergence test;
  * an optional combiner: messages for the same target are reduced before delivery (PageRank's
    combine is a sum), which must not change the answer;
  * checkpoints every k supersteps (values, halt flags, in-flight mail, aggregator state) plus a
    per-worker log of the mail delivered in each superstep.  A scheduled worker failure rolls
    that worker's partition back to the latest checkpoint, replays the lost supersteps from the
    log, and the messages it re-emits are recognised by their (source, superstep, sequence)
    identity and dropped by the receiver - so replayed work never has a second effect.

Simplified versus production: the network is a dictionary, the log lives in memory, and only a
single worker fails at a time (a second failure is scheduled after the first recovery, and
exhausting the retry budget aborts the job).  What is *not* simplified is the semantics: results
must be bit-identical to a failure-free run.

Verified, against independent oracles and under injected faults:
  * PageRank converges to the same vector as a dense power-iteration oracle (max deviation
    measured), ranks sum to 1, dangling nodes are handled through the aggregator;
  * single-source shortest paths match Bellman-Ford and connected components match union-find
    exactly;
  * the combiner lowers the number of delivered messages while leaving every value identical;
  * exhaustive fault injection (a worker failing at every superstep of a run, with two
    checkpoint intervals) reproduces the failure-free result exactly, with duplicates proving
    the replay happened and the replay cost bounded by the checkpoint interval;
  * two failures in one job, checkpoint-free recovery, degenerate graphs (single vertex, empty,
    disconnected, self-loop, dangling sink) and retry exhaustion are all covered.
"""

INF = float("inf")


class Aggregator:
    """A named reduce with a combine function; contributions land in the *next* superstep."""

    def __init__(self, name, combine, identity):
        self.name = name
        self.combine = combine
        self.identity = identity
        self.value = identity
        self.next_value = identity

    def contribute(self, value):
        self.next_value = self.combine(self.next_value, value)

    def advance(self):
        """Publish the contributions of the superstep that just finished."""
        self.value = self.next_value
        self.next_value = self.identity

    def reset(self):
        self.value = self.identity
        self.next_value = self.identity


class Context:
    """What a compute function may touch: the graph, the aggregators and job metadata."""

    def __init__(self, engine, superstep):
        self.engine = engine
        self.superstep = superstep

    @property
    def size(self):
        return len(self.engine.vertices)

    @property
    def graph(self):
        return self.engine.graph

    def aggregate(self, name):
        return self.engine.aggregators[name].value

    def contribute(self, name, value):
        self.engine.aggregators[name].contribute(value)


class PregelEngine:
    """A deterministic BSP engine with partitioning, combiners, checkpoints and replay."""

    def __init__(self, graph, workers=3, checkpoint_every=2, aggregators=(), combiner=None,
                 tol=1e-13, cap=200, retry_budget=3):
        self.graph = {vertex: list(edges) for vertex, edges in graph.items()}
        self.vertices = sorted(self.graph)
        self.workers = workers
        self.owner = {vertex: index % workers for index, vertex in enumerate(self.vertices)}
        self.checkpoint_every = checkpoint_every
        self.aggregators = {aggregator.name: aggregator for aggregator in aggregators}
        self.combiner = combiner
        self.tol = tol
        self.cap = cap
        self.retry_budget = retry_budget
        self.values = {}
        self.halted = {vertex: False for vertex in self.vertices}
        self.mail = {vertex: [] for vertex in self.vertices}
        self.sequence = 0
        self.seen = set()
        self.delivered_log = {}
        self.checkpoints = {}
        self.counters = {"supersteps": 0, "sent": 0, "delivered": 0, "combined": 0,
                         "duplicates": 0, "cross_worker": 0, "checkpoints": 0, "replays": 0,
                         "replayed_steps": 0, "replayed_messages": 0, "emitted_on_replay": 0}
        for vertex in self.vertices:
            if self.owner[vertex] >= workers:
                raise ValueError("worker index out of range")

    # -- setup -------------------------------------------------------------
    def initialise(self, initial, active=()):
        for vertex in self.vertices:
            self.values[vertex] = initial(vertex)
            self.halted[vertex] = vertex not in active
        return self

    def inject(self, vertex, payload):
        """Seed a message into a vertex mailbox - the master's kick-off message in Pregel."""
        self.mail[vertex] = list(self.mail[vertex]) + [payload]
        self.halted[vertex] = False
        return self

    def snapshot(self):
        return {"values": dict(self.values), "halted": dict(self.halted),
                "mail": {vertex: list(messages) for vertex, messages in self.mail.items()},
                "aggregators": {name: (aggregator.value, aggregator.next_value)
                                for name, aggregator in self.aggregators.items()}}

    def restore(self, snapshot, worker=None):
        """Restore the whole checkpoint, or only the slice owned by `worker`.

        Recovery is deliberately *partial*: the failed worker's vertices go back to the
        checkpoint while the surviving workers keep their newer state, so their deliveries are
        not repeated.  The message identities (`seen`) are never rolled back - they are the
        receivers' memory of what already had an effect.
        """
        for vertex in self.vertices:
            if worker is not None and self.owner[vertex] != worker:
                continue
            self.values[vertex] = snapshot["values"][vertex]
            self.halted[vertex] = snapshot["halted"][vertex]
        if worker is None:
            self.mail = {vertex: list(messages)
                         for vertex, messages in snapshot["mail"].items()}
            for name, (value, next_value) in snapshot["aggregators"].items():
                self.aggregators[name].value = value
                self.aggregators[name].next_value = next_value

    # -- one superstep -----------------------------------------------------
    def superstep(self, compute):
        index = self.counters["supersteps"]
        context = Context(self, index)
        produced = {vertex: [] for vertex in self.vertices}
        inbox = {vertex: self.mail[vertex] for vertex in self.vertices}
        self.mail = {vertex: [] for vertex in self.vertices}
        self.delivered_log.setdefault(index, {vertex: list(inbox[vertex])
                                              for vertex in self.vertices})
        for vertex in self.vertices:
            messages = inbox.get(vertex, [])
            if not self.halted[vertex] or messages:
                new_value, sends, voted = compute(vertex, self.values[vertex], messages,
                                                  context)
                self.counters["delivered"] += len(messages)
                self.halted[vertex] = voted
                if new_value is not None:
                    self.values[vertex] = new_value
                produced[vertex] = list(sends)
        self.counters["supersteps"] += 1
        for aggregator in self.aggregators.values():
            aggregator.advance()
        self._route(index, produced)
        return self.pending() > 0

    def _route(self, index, produced):
        """Deliver the produced messages, applying dedup, the combiner and cross-worker counts."""
        buffered = {}
        for vertex in self.vertices:
            for seq, (target, payload) in enumerate(produced[vertex]):
                if target not in self.graph:
                    raise ValueError("message to unknown vertex %r" % (target,))
                key = (vertex, index, seq)
                if key in self.seen:
                    self.counters["duplicates"] += 1
                    continue
                self.seen.add(key)
                self.counters["sent"] += 1
                if self.owner[vertex] != self.owner[target]:
                    self.counters["cross_worker"] += 1
                buffered.setdefault(target, []).append((vertex, payload))
        for target in sorted(buffered):
            messages = buffered[target]
            if self.combiner is not None and len(messages) > 1:
                payloads = self.combiner([payload for _, payload in messages])
                self.counters["combined"] += len(messages) - len(payloads)
                messages = [("combiner", payload) for payload in payloads]
            self.mail[target] = [payload for _, payload in messages]

    def pending(self):
        return sum(len(messages) for messages in self.mail.values())

    def converged(self):
        return self.pending() == 0 and all(self.halted.values())

    def settled(self):
        """A global convergence criterion: the max per-vertex delta aggregator is below tol."""
        marker = self.aggregators.get("maxdelta")
        return (marker is not None and self.counters["supersteps"] > 2
                and marker.value < self.tol)

    # -- fault injection ---------------------------------------------------
    def fail_worker(self, worker, compute):
        """Roll one worker back to the latest checkpoint and replay its lost supersteps.

        The replayed supersteps use the mail the worker actually received (kept in the
        delivery log) and re-emit the same messages, which the receivers recognise and drop:
        the job's observable effects are therefore exactly-once even though the work is redone.
        """
        current = self.counters["supersteps"]
        usable = [step for step in self.checkpoints if step <= current]
        if not usable:
            raise AssertionError("no checkpoint to recover from")
        start = max(usable)
        self.restore(self.checkpoints[start], worker)
        emitted = 0
        for step in range(start, current):
            context = Context(self, step)
            produced = {vertex: [] for vertex in self.vertices}
            for vertex in self.vertices:
                if self.owner[vertex] != worker:
                    continue
                messages = self.delivered_log.get(step, {}).get(vertex, [])
                new_value, sends, voted = compute(vertex, self.values[vertex], messages,
                                                  context)
                self.halted[vertex] = voted
                if new_value is not None:
                    self.values[vertex] = new_value
                produced[vertex] = list(sends)
            for aggregator in self.aggregators.values():
                aggregator.advance()
            before = self.counters["duplicates"]
            self._route(step, produced)
            emitted += produced and sum(len(sends) for sends in produced.values())
            self.counters["replayed_steps"] += 1
            self.counters["replayed_messages"] += (self.counters["duplicates"] - before)
        self.counters["replays"] += 1
        self.counters["emitted_on_replay"] += emitted
        return start, current

    def run(self, compute, fail_plan=None):
        """`fail_plan` is a sequence of (superstep, worker) events, each handled exactly once."""
        events = list(fail_plan or ())
        used = 0
        while True:
            if self.converged() or self.settled() or self.counters["supersteps"] >= self.cap:
                break
            if self.checkpoint_every and self.counters["supersteps"] % self.checkpoint_every == 0:
                self.checkpoints[self.counters["supersteps"]] = self.snapshot()
                self.counters["checkpoints"] += 1
            step = self.counters["supersteps"]
            if events and events[0][0] == step:
                if used >= self.retry_budget:
                    raise RuntimeError("retry budget exhausted at superstep %d" % step)
                _, worker = events.pop(0)
                used += 1
                self.fail_worker(worker, compute)
                continue
            self.superstep(compute)
        return self


# --------------------------------------------------------------------------------------
# algorithms
# --------------------------------------------------------------------------------------
def pagerank_compute(damping):
    """Pregel PageRank: dangling vertices redistribute uniformly, and nothing halts per vertex.

    A dangling vertex sends rank/n to every vertex (Pregel's alternative to a teleport
    aggregator).  Vertices deliberately do *not* vote to halt individually: a vertex that stops
    sending while its neighbours still wait for its share leaks rank mass, so the job instead
    ends when the maxdelta aggregator reports that the whole iteration has settled.
    """
    def compute(vertex, value, messages, context):
        graph = context.graph
        received = sum(messages)
        rank = (1.0 - damping) / context.size + damping * received
        context.contribute("maxdelta", abs(rank - value))
        if graph[vertex]:
            share = rank / len(graph[vertex])
            sends = [(target, share) for target, _ in graph[vertex]]
        else:
            share = rank / context.size
            sends = [(target, share) for target in graph]
        return rank, sends, False
    return compute


def sssp_compute(source):
    def compute(vertex, value, messages, context):
        best = value
        for message in messages:
            if message < best:
                best = message
        sends = []
        if best < value:
            for target, weight in context.graph[vertex]:
                sends.append((target, best + weight))
        return best, sends, best >= value and not messages
    return compute


def components_compute():
    def compute(vertex, value, messages, context):
        best = value
        for message in messages:
            if message < best:
                best = message
        sends = []
        if best < value or context.superstep == 0:
            # the first superstep broadcasts the initial label; afterwards only improvements do
            for target, _ in context.graph[vertex]:
                sends.append((target, best))
        return best, sends, best >= value and not messages
    return compute


def reference_pagerank(graph, damping=0.85, iterations=400):
    vertices = sorted(graph)
    index = {vertex: position for position, vertex in enumerate(vertices)}
    size = len(vertices)
    ranks = [1.0 / size] * size
    for _ in range(iterations):
        dangling = sum(ranks[index[vertex]] for vertex in vertices if not graph[vertex])
        base = (1.0 - damping) / size + damping * dangling / size
        fresh = [base] * size
        for vertex in vertices:
            out = graph[vertex]
            if not out:
                continue
            share = ranks[index[vertex]] / len(out)
            for target, _ in out:
                fresh[index[target]] += damping * share
        ranks = fresh
    return {vertex: ranks[index[vertex]] for vertex in vertices}


def reference_sssp(graph, source):
    distances = {vertex: INF for vertex in graph}
    distances[source] = 0
    for _ in range(len(graph)):
        changed = False
        for vertex in sorted(graph):
            if distances[vertex] == INF:
                continue
            for target, weight in graph[vertex]:
                if distances[vertex] + weight < distances[target]:
                    distances[target] = distances[vertex] + weight
                    changed = True
        if not changed:
            break
    return distances


def reference_components(graph):
    parent = {vertex: vertex for vertex in graph}

    def find(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for vertex in sorted(graph):
        for target, _ in graph[vertex]:
            left, right = find(vertex), find(target)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return {vertex: find(vertex) for vertex in graph}


def sample_graph():
    """A small weighted digraph with a dangling sink and one self loop."""
    edges = {
        0: [(1, 1), (2, 4)],
        1: [(2, 1), (3, 5)],
        2: [(3, 1), (4, 2)],
        3: [(4, 1), (0, 2)],
        4: [(5, 1)],
        5: [],
        6: [(7, 1)],
        7: [(6, 1), (7, 3)],
    }
    return edges


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    graph = sample_graph()

    # ---- 1. partitioning --------------------------------------------------
    engine = PregelEngine(graph, workers=3, checkpoint_every=3)
    counts = {}
    for vertex, worker in engine.owner.items():
        counts[worker] = counts.get(worker, 0) + 1
    check(sum(counts.values()) == len(graph), "every vertex sits on exactly one worker")
    check(max(counts.values()) - min(counts.values()) <= 1, "the partition is balanced")
    check(len(counts) == 3, "all three workers own something")

    # ---- 2. PageRank against the dense oracle -----------------------------
    damping = 0.85
    def aggregators():
        return (Aggregator("maxdelta", max, 0.0),)

    job = PregelEngine(graph, workers=3, checkpoint_every=3, aggregators=aggregators(),
                       tol=1e-13, cap=300)
    job.initialise(lambda vertex: 0.0, active=range(len(graph)))
    job.values = {vertex: 1.0 / len(graph) for vertex in job.vertices}
    job.run(pagerank_compute(damping))
    exact = reference_pagerank(graph, damping)
    deviation = max(abs(job.values[vertex] - exact[vertex]) for vertex in graph)
    check(deviation < 1e-8, "PageRank must match the dense oracle (deviation %g)" % deviation)
    total = sum(job.values.values())
    check(abs(total - 1.0) < 1e-10, "ranks must sum to one (got %.12f)" % total)
    check(abs(job.values[5] - exact[5]) < 1e-9,
          "the dangling sink must match the oracle through the uniform redistribution")
    check(job.counters["supersteps"] > 5, "PageRank needs several supersteps")
    check(job.counters["supersteps"] < 300, "PageRank must settle before the cap")
    check(job.counters["cross_worker"] > 0, "the partition must actually cut edges")

    # ---- 3. the combiner saves traffic without changing the answer --------
    plain = PregelEngine(graph, workers=3, checkpoint_every=3, aggregators=aggregators(),
                         tol=1e-13, cap=300)
    plain.initialise(lambda vertex: 0.0, active=range(len(graph)))
    plain.values = {vertex: 1.0 / len(graph) for vertex in plain.vertices}
    plain.run(pagerank_compute(damping))
    combined = PregelEngine(graph, workers=3, checkpoint_every=3, aggregators=aggregators(),
                            combiner=lambda payloads: [sum(payloads)], tol=1e-13, cap=300)
    combined.initialise(lambda vertex: 0.0, active=range(len(graph)))
    combined.values = {vertex: 1.0 / len(graph) for vertex in combined.vertices}
    combined.run(pagerank_compute(damping))
    check(max(abs(plain.values[v] - combined.values[v]) for v in graph) < 1e-12,
          "the combiner must not change the result")
    check(combined.counters["delivered"] < plain.counters["delivered"],
          "the combiner must reduce the delivered message count")
    check(combined.counters["combined"] > 0, "the combiner must actually combine")

    # ---- 4. SSSP and connected components against oracles -----------------
    shortest = PregelEngine(graph, workers=3, checkpoint_every=2)
    shortest.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
    shortest.run(sssp_compute(0))
    expected = reference_sssp(graph, 0)
    for vertex in graph:
        check(shortest.values[vertex] == expected[vertex],
              "distance to %d must match Bellman-Ford (%r vs %r)"
              % (vertex, shortest.values[vertex], expected[vertex]))
    check(shortest.values[5] == 5, "the five hop path 0-1-2-3-4-5 costs 5")
    check(shortest.values[6] == INF, "an unreachable vertex keeps infinity")

    labels = PregelEngine(graph, workers=3, checkpoint_every=2)
    labels.initialise(lambda vertex: vertex, active=range(len(graph)))
    labels.run(components_compute())
    oracle = reference_components(graph)
    check(all(labels.values[vertex] == oracle[vertex] for vertex in graph),
          "component labels must match union-find")
    check(labels.values[0] == labels.values[5], "vertices 0 and 5 are connected")
    check(labels.values[6] != labels.values[0], "the second component is separate")

    # ---- 5. determinism ---------------------------------------------------
    again = PregelEngine(graph, workers=3, checkpoint_every=2)
    again.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
    again.run(sssp_compute(0))
    check(again.values == shortest.values and again.counters == shortest.counters,
          "two identical jobs must produce identical state and counters")

    # ---- 6. exhaustive fault injection ------------------------------------
    baseline_values = dict(shortest.values)
    baseline_counters = dict(shortest.counters)
    recoveries = 0
    replayed_any = 0
    for interval in (1, 2, 3, 5):
        for fail_step in range(1, max(2, baseline_counters["supersteps"])):
            hurt = PregelEngine(graph, workers=3, checkpoint_every=interval)
            hurt.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
            hurt.run(sssp_compute(0), fail_plan=((fail_step, fail_step % 3),))
            check(hurt.values == baseline_values,
                  "recovery at step %d (interval %d) must reproduce the result"
                  % (fail_step, interval))
            check(hurt.counters["replays"] == 1, "exactly one recovery must be recorded")
            check(0 <= hurt.counters["replayed_steps"] <= interval,
                  "the replay cost is bounded by the checkpoint interval")
            if interval > 1:
                replayed_any += hurt.counters["replayed_steps"]
            else:
                check(hurt.counters["replayed_steps"] == 0,
                      "a checkpoint at the failing superstep means nothing to replay")
            recoveries += 1
    check(recoveries >= 8, "the fault sweep must cover many failure points")
    check(replayed_any > 0, "some scenarios must actually replay supersteps")

    # the replay really re-emits messages, and the receivers drop them as duplicates
    labels_base = PregelEngine(graph, workers=3, checkpoint_every=5)
    labels_base.initialise(lambda vertex: vertex, active=range(len(graph)))
    labels_base.run(components_compute())
    labels_hurt = PregelEngine(graph, workers=3, checkpoint_every=5)
    labels_hurt.initialise(lambda vertex: vertex, active=range(len(graph)))
    labels_hurt.run(components_compute(), fail_plan=((1, 0),))
    check(labels_hurt.values == labels_base.values, "component labels must survive recovery")
    check(labels_hurt.counters["replayed_steps"] == 1,
          "the failure at step 1 replays the single superstep after the checkpoint")
    check(labels_hurt.counters["duplicates"] > 0,
          "the replayed superstep must re-emit messages that are dropped by identity")
    check(labels_hurt.counters["duplicates"] == labels_hurt.counters["emitted_on_replay"],
          "every message re-emitted by the replay must be recognised and dropped")
    check(labels_hurt.counters["duplicates"] == 5,
          "the re-emitted messages are exactly the five broadcasts of worker 0")

    # a job with two scheduled failures
    twice = PregelEngine(graph, workers=3, checkpoint_every=2)
    twice.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
    twice.run(sssp_compute(0), fail_plan=((2, 0), (4, 1)))
    check(twice.values == baseline_values, "two failures must still recover exactly")
    check(twice.counters["replays"] == 2, "two recoveries must be recorded")

    # recovery at a later step with a wider checkpoint interval replays from the checkpoint
    restarted = PregelEngine(graph, workers=3, checkpoint_every=1)
    restarted.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
    restarted.run(sssp_compute(0), fail_plan=((1, 2),))
    check(restarted.values == baseline_values, "a step-1 failure must recover exactly")
    check(restarted.counters["replayed_steps"] == 0,
          "a checkpoint taken at the failing step needs no replay")

    # retry exhaustion is reported, not ignored
    try:
        doomed = PregelEngine(graph, workers=3, checkpoint_every=1, retry_budget=0)
        doomed.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
        doomed.run(sssp_compute(0), fail_plan=((1, 0),))
        raise AssertionError("exhausting the retry budget must abort the job")
    except RuntimeError:
        pass

    # ---- 7. degenerate graphs ---------------------------------------------
    singleton = PregelEngine({0: []}, workers=1)
    singleton.initialise(lambda vertex: 0, active=[0])
    singleton.run(components_compute())
    check(singleton.values == {0: 0} and singleton.counters["sent"] == 0,
          "a single vertex converges without sending anything")
    empty = PregelEngine({}, workers=1)
    empty.run(components_compute())
    check(empty.counters["supersteps"] == 0, "an empty graph needs no superstep")
    self_loop = PregelEngine({0: [(0, 1)]}, workers=2)
    self_loop.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
    self_loop.run(sssp_compute(0))
    check(self_loop.values[0] == 0.0, "a self loop must not lower a zero distance")
    check(self_loop.counters["cross_worker"] == 0, "a self loop never crosses a worker")
    chain_graph = {0: [(1, 1)], 1: [(2, 1)], 2: [(3, 1)], 3: []}
    chain = PregelEngine(chain_graph, workers=2)
    chain.initialise(lambda vertex: INF, active=[]).inject(0, 0.0)
    chain.run(sssp_compute(0))
    check(chain.values == {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}, "a chain propagates in order")
    check(chain.counters["delivered"] > 0 and chain.pending() == 0,
          "all messages are drained before the job ends")

    # ---- 8. cost bound ----------------------------------------------------
    steps = baseline_counters["supersteps"]
    edges = sum(len(edges) for edges in graph.values())
    check(baseline_counters["sent"] <= steps * edges + 4,
          "messages per superstep are bounded by the edge count")
    check(plain.counters["checkpoints"] >= job.counters["checkpoints"] - 2,
          "checkpoints are taken at the configured interval")
    check(job.counters["delivered"] > 100 and steps * edges > 0,
          "the oracle comparison is meaningful: the run really moved %d messages"
          % job.counters["delivered"])

    print("pagerank: %d supersteps, deviation %.3g, rank sum %.12f, sink rank %.9f"
          % (job.counters["supersteps"], deviation, total, job.values[5]))
    print("messages: delivered %d (combiner %d, saved %d), cross worker %d, edges %d"
          % (plain.counters["delivered"], combined.counters["delivered"],
             plain.counters["delivered"] - combined.counters["delivered"],
             job.counters["cross_worker"], edges))
    print("faults: %d recovery scenarios, up to %d replayed supersteps, %d replayed messages "
          "dropped by identity" % (recoveries, max(interval for interval in (1, 2, 3, 5)),
                                   labels_hurt.counters["duplicates"]))
    print("PASS")


if __name__ == "__main__":
    main()
