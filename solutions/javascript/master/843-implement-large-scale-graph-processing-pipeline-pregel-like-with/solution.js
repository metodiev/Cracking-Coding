/** Task 843 - Implement large-scale graph processing pipeline (Pregel-like) with fault tolerance.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * A Pregel-style bulk synchronous parallel (BSP) engine. The graph is split over workers, every
 * vertex runs the same program, and vertices communicate *only* through messages that cross an
 * explicit in-memory network between supersteps:
 *
 *  - barriers: all vertices of a superstep observe only the previous superstep's messages, so no
 *    vertex can ever see a message produced in its own superstep (checked directly by a probe
 *    program that records the superstep of every message it receives);
 *  - halting: a vertex votes to halt and is woken again by an incoming message; the run ends when
 *    every vertex is halted and no message is in flight;
 *  - combiners: messages addressed to the same vertex inside one superstep are merged by an
 *    algorithm-supplied associative reducer before routing, which reduces delivered traffic without
 *    changing the result;
 *  - aggregators: global reduce-and-broadcast values (sum / min / max) recomputed at every barrier,
 *    used here for the frontier size, the smallest label and PageRank's dangling mass;
 *  - fault tolerance: every `checkpointEvery` supersteps the master snapshots all worker states, and
 *    on a worker failure it rolls the whole cluster back to the last checkpoint and re-executes
 *    (Pregel's coordinated rollback, with the sender rollback modelled as a global one). A
 *    deterministic engine must then land on *exactly* the failure-free state, and the replayed
 *    supersteps / re-sent messages are reported as the cost of recovery.
 *
 * Programs: single-source shortest paths, connected components (min-label propagation) and PageRank
 * with dangling-mass redistribution, each cross-checked against an independent oracle (Dijkstra,
 * union-find, dense power iteration).
 *
 * Simplifications, stated honestly: workers are in-process objects stepped one at a time in a sorted
 * order instead of threads; "scale" is a parameter (vertex count, partition count, edge density);
 * checkpoints are in-memory snapshots with a digest rather than distributed storage; stragglers and
 * timeouts are not modelled; the run length of iterative programs is bounded explicitly. No timers,
 * threads or sockets — only seeded splitmix64 for randomness.
 *
 * Invariants verified below:
 *  - safety: no same-superstep delivery, aggregators reset per superstep, halting only undone by a
 *    message, and the run stops exactly when everything is halted and the network is empty;
 *  - correctness against oracles: SSSP = Dijkstra on every vertex, components = union-find labels,
 *    PageRank = power iteration within 1e-5 with ranks summing to 1;
 *  - liveness: every algorithm halts inside a diameter/compression bound (measured superstep counts),
 *    including empty, single-vertex, disconnected and self-loop graphs;
 *  - fault tolerance: with two injected worker failures the recovered state digest equals the
 *    failure-free digest, no rollback exceeds the checkpoint interval, and a variant that skips
 *    recovery demonstrably produces a different (wrong) answer;
 *  - cost: message conservation (sent = delivered + combined), local vs cross-partition traffic, and
 *    the replayed superstep/message counts.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const INFINITY = Infinity;

/** Deterministic splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = (BigInt(seed) & MASK64) || 1n;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  upto(bound) {
    return Number(this.next() % BigInt(bound));
  }
}

/** Deterministic rolling digest over vertex values, used for checkpoints and run comparison. */
function digest(value) {
  const text = JSON.stringify(value, (key, item) => (item === INFINITY ? "Infinity" : item));
  let h = 0xcbf29ce484222325n;
  for (let i = 0; i < text.length; i++) {
    h = (h ^ BigInt(text.charCodeAt(i))) & MASK64;
    h = (h * 0x100000001b3n) & MASK64;
  }
  return h.toString(16).padStart(16, "0");
}

/** Deterministic graph: a spanning path for connectivity plus `extraEdges` random arcs. */
function buildGraph(options = {}) {
  const vertices = options.vertices ?? 24;
  const extraEdges = options.extraEdges ?? 20;
  const seed = options.seed ?? 843;
  const directed = options.directed ?? true;
  const weighted = options.weighted ?? true;
  const connected = options.connected ?? true;
  const rng = new SplitMix64(seed);
  const adjacency = new Map();
  for (let i = 0; i < vertices; i++) adjacency.set(i, []);
  const addEdge = (from, to, weight) => {
    adjacency.get(from).push({ to, weight });
    if (!directed) adjacency.get(to).push({ to: from, weight });
  };
  for (let i = 1; i < vertices && connected; i++) addEdge(i - 1, i, weighted ? 1 + rng.upto(9) : 1);
  for (let i = 0; i < extraEdges; i++) {
    const from = rng.upto(vertices);
    const to = rng.upto(vertices);
    if (from === to) continue;
    addEdge(from, to, weighted ? 1 + rng.upto(9) : 1);
  }
  return { adjacency, vertices, directed, seed };
}

/** Round-robin partitioning of the sorted vertex ids over `partitions` workers. */
function assignPartitions(graph, partitions) {
  const ids = [...graph.adjacency.keys()].sort((a, b) => a - b);
  const partitionOf = new Map();
  ids.forEach((id, index) => partitionOf.set(id, index % partitions));
  return partitionOf;
}

const REDUCERS = {
  min: (a, b) => (b < a ? b : a),
  max: (a, b) => (b > a ? b : a),
  sum: (a, b) => a + b,
};

/** The part of a vertex that must survive a checkpoint. */
class VertexState {
  constructor(id, value) {
    this.id = id;
    this.value = value;
    this.halted = false;
    this.inbox = [];
  }

  snapshot() {
    return { id: this.id, value: this.value, halted: this.halted, inbox: this.inbox.slice() };
  }
}

/**
 * The Pregel engine: vertices live on workers, messages cross the network at the barrier,
 * combiners/aggregators are applied at that barrier, and checkpoints enable coordinated rollback.
 */
class PregelEngine {
  constructor(graph, options = {}) {
    this.graph = graph;
    this.partitions = options.partitions ?? 4;
    this.partitionOf = assignPartitions(graph, this.partitions);
    this.combiner = options.combiner ?? null;
    this.aggregatorDefs = options.aggregators ?? {};
    this.checkpointEvery = options.checkpointEvery ?? 3;
    this.recovery = options.recovery ?? "rollback";
    this.reset(null);
  }

  /** Creates the initial vertex states; `initialValue` may be a function (id, graph) -> value. */
  reset(initialValue) {
    this.states = new Map();
    for (const id of this.graph.adjacency.keys()) {
      const value = typeof initialValue === "function" ? initialValue(id, this.graph) : initialValue;
      this.states.set(id, new VertexState(id, value));
    }
    this.superstep = 0;
    this.messagesSent = 0;
    this.messagesDelivered = 0;
    this.messagesCombined = 0;
    this.messagesReplayed = 0;
    this.localMessages = 0;
    this.crossPartitionMessages = 0;
    this.checkpoints = [];
    this.faultEvents = [];
    this.replayedSupersteps = 0;
    this.aggregateLog = [];
    this.lastAggregates = new Map();
    this.replaying = false;
  }

  /** Snapshot of every vertex state: the recovery point. */
  checkpoint() {
    const snapshot = {
      superstep: this.superstep,
      states: [...this.states.values()].map((state) => state.snapshot()),
    };
    snapshot.digest = digest(snapshot.states
      .map((state) => [state.id, state.value, state.halted, state.inbox.length]));
    this.checkpoints.push(snapshot);
    return snapshot;
  }

  restore(snapshot) {
    this.states = new Map();
    for (const state of snapshot.states) {
      const restored = new VertexState(state.id, state.value);
      restored.halted = state.halted;
      restored.inbox = state.inbox.slice();
      this.states.set(state.id, restored);
    }
    this.superstep = snapshot.superstep;
  }

  /** One superstep: compute on every worker, then combine, deliver and aggregate at the barrier. */
  step(program, initialValue) {
    const outgoing = new Map(); // target -> [message]
    const aggregates = new Map();
    for (const [name, def] of Object.entries(this.aggregatorDefs)) {
      aggregates.set(name, def.initial);
    }
    let liveVertices = 0;
    for (const id of [...this.states.keys()].sort((a, b) => a - b)) {
      const state = this.states.get(id);
      const inbox = state.inbox;
      state.inbox = [];
      const send = (target, message) => {
        if (!outgoing.has(target)) outgoing.set(target, []);
        outgoing.get(target).push(message);
        this.messagesSent++;
        if (this.replaying) this.messagesReplayed++;
        if (this.partitionOf.get(target) === this.partitionOf.get(id)) this.localMessages++;
        else this.crossPartitionMessages++;
      };
      const context = {
        superstep: this.superstep,
        vertexId: id,
        value: state.value,
        edges: this.graph.adjacency.get(id) ?? [],
        send,
        aggregate: (name, value) => {
          const def = this.aggregatorDefs[name];
          if (def === undefined) throw new Error(`unknown aggregator ${name}`);
          aggregates.set(name, def.reduce(aggregates.get(name), value));
        },
        aggregateOf: (name) => this.lastAggregates.get(name),
        voteToHalt: () => {
          state.halted = true;
        },
      };
      if (!state.halted || inbox.length > 0) {
        state.halted = false; // a delivered message wakes a halted vertex, exactly like Pregel
        liveVertices++;
        program.compute(context, inbox, {
          setValue: (value) => {
            state.value = value;
          },
          self: state,
        });
      }
    }
    // Barrier: combine per target, then deliver into the next inboxes.
    for (const [target, list] of outgoing) {
      let payload = list;
      if (this.combiner !== null && list.length > 1) {
        let accumulator = list[0];
        for (let i = 1; i < list.length; i++) {
          accumulator = this.combiner.reduce(accumulator, list[i]);
        }
        payload = [accumulator];
        this.messagesCombined += list.length - 1;
      }
      const state = this.states.get(target);
      if (state === undefined) continue; // a message to an unknown vertex is dropped
      state.inbox.push(...payload);
      this.messagesDelivered += payload.length;
    }
    this.lastAggregates = aggregates;
    this.aggregateLog.push(Object.fromEntries([...aggregates].map(([name, value]) => [name, value])));
    this.superstep++;
    return { liveVertices, aggregates };
  }

  /** True when every vertex is halted and no message is in flight. */
  finished() {
    for (const state of this.states.values()) {
      if (!state.halted || state.inbox.length > 0) return false;
    }
    return true;
  }

  stateDigest() {
    return digest([...this.states.keys()].sort((a, b) => a - b)
      .map((id) => [id, this.states.get(id).value, this.states.get(id).halted]));
  }

  valueOf(id) {
    return this.states.get(id).value;
  }

  /**
   * Runs to completion, injecting the requested worker faults (each fires once). A fault at superstep
   * s rolls the cluster back to the last checkpoint; the recovery mode "none" models an engine that
   * simply wipes the failed worker and carries on, which must be visibly wrong.
   */
  run(program, options = {}) {
    const maxSupersteps = options.maxSupersteps ?? 200;
    const initialValue = options.initialValue ?? (() => 0);
    const faults = options.faults ?? [];
    const fired = new Set();
    let latest = null;
    while (this.superstep < maxSupersteps) {
      const fault = faults.find((candidate) => candidate.superstep === this.superstep &&
        !fired.has(candidate));
      if (fault !== undefined) {
        fired.add(fault);
        this.faultEvents.push({ superstep: this.superstep, worker: fault.worker });
        if (this.recovery === "rollback" && latest !== null) {
          const delta = this.superstep - latest.superstep; // measure before rolling back
          this.restore(latest);
          this.replayedSupersteps += delta;
          this.replaying = true;
        } else if (this.recovery === "none") {
          for (const [id, state] of this.states) {
            if (this.partitionOf.get(id) !== fault.worker) continue;
            state.value = initialValue(id, this.graph);
            state.inbox = [];
            state.halted = false;
          }
        }
      } else {
        this.replaying = false;
      }
      this.step(program, initialValue);
      if (this.superstep % this.checkpointEvery === 0 || this.finished()) latest = this.checkpoint();
      if (this.finished()) break;
    }
    this.replaying = false;
    return { supersteps: this.superstep, checkpoints: this.checkpoints.length,
      halted: this.finished() };
  }
}

// --- programs ---------------------------------------------------------------
/** SSSP: relax along out-edges, send only on improvement, combine parallel relaxations with min. */
const ssspProgram = {
  name: "sssp",
  initial: (id, graph, source) => (id === source ? 0 : INFINITY),
  combiner: { name: "min", reduce: REDUCERS.min },
  aggregators: {
    frontier: { reduce: REDUCERS.sum, initial: 0 },
    minDistance: { reduce: REDUCERS.min, initial: INFINITY },
  },
  compute(context, inbox, sink) {
    let best = context.value;
    for (const message of inbox) if (message < best) best = message;
    context.aggregate("frontier", inbox.length);
    context.aggregate("minDistance", best);
    if (best < context.value) {
      sink.setValue(best);
      for (const edge of context.edges) context.send(edge.to, best + edge.weight);
    } else if (context.superstep === 0 && context.value === 0) {
      // The source seeds its own neighbours in the first superstep.
      for (const edge of context.edges) context.send(edge.to, edge.weight);
    }
    context.voteToHalt();
  },
};

/** Connected components: propagate the smallest label seen, combine with min. */
const componentsProgram = {
  name: "components",
  initial: (id) => id,
  combiner: { name: "min", reduce: REDUCERS.min },
  aggregators: {
    changed: { reduce: REDUCERS.sum, initial: 0 },
    smallestLabel: { reduce: REDUCERS.min, initial: Infinity },
  },
  compute(context, inbox, sink) {
    let smallest = context.value;
    for (const message of inbox) if (message < smallest) smallest = message;
    context.aggregate("smallestLabel", smallest);
    if (smallest < context.value) {
      sink.setValue(smallest);
      context.aggregate("changed", 1);
      for (const edge of context.edges) context.send(edge.to, smallest);
    } else if (context.superstep === 0) {
      for (const edge of context.edges) context.send(edge.to, context.value);
    }
    context.voteToHalt();
  },
};

/** PageRank: push rank/outDegree along out-edges; dangling mass is redistributed via an aggregator. */
const pageRankProgram = (damping, urlCount) => ({
  name: "pagerank",
  initial: () => 1 / urlCount,
  combiner: { name: "sum", reduce: REDUCERS.sum },
  aggregators: {
    dangling: { reduce: REDUCERS.sum, initial: 0 },
    totalRank: { reduce: REDUCERS.sum, initial: 0 },
  },
  compute(context, inbox, sink) {
    let incoming = 0;
    for (const message of inbox) incoming += message;
    if (context.superstep === 0) incoming = 1 / urlCount; // the initial uniform distribution
    incoming += context.aggregateOf("dangling") ?? 0;
    const outDegree = context.edges.length;
    const rank = (1 - damping) / urlCount + damping * incoming;
    sink.setValue(rank);
    context.aggregate("totalRank", rank);
    if (outDegree === 0) context.aggregate("dangling", rank / urlCount);
    const share = outDegree === 0 ? 0 : rank / outDegree; // the receiver applies the damping factor
    for (const edge of context.edges) context.send(edge.to, share);
  },
});

/** Probe program: records the superstep of every message, flagging any same-superstep delivery. */
const superstepProbeProgram = {
  name: "probe",
  initial: () => 0,
  compute(context, inbox, sink) {
    if (context.superstep === 0) {
      for (const edge of context.edges) context.send(edge.to, context.superstep);
      sink.setValue(context.value + 1);
    } else {
      let seen = context.value;
      for (const message of inbox) {
        if (message >= context.superstep) context.self.barrierViolation = true;
        seen += 1 + message;
      }
      sink.setValue(seen);
    }
    context.voteToHalt();
  },
};

// --- independent oracles ----------------------------------------------------
/** Dijkstra with a linear scan: independent of the engine and its message plumbing. */
function dijkstra(graph, source) {
  const distance = new Map();
  for (const id of graph.adjacency.keys()) distance.set(id, INFINITY);
  distance.set(source, 0);
  const settled = new Set();
  for (;;) {
    let best = null;
    for (const [id, value] of distance) {
      if (settled.has(id) || value === INFINITY) continue;
      if (best === null || value < distance.get(best)) best = id;
    }
    if (best === null) break;
    settled.add(best);
    for (const edge of graph.adjacency.get(best)) {
      const candidate = distance.get(best) + edge.weight;
      if (candidate < distance.get(edge.to)) distance.set(edge.to, candidate);
    }
  }
  return distance;
}

/** Union-find over the same adjacency lists. */
function unionFindComponents(graph) {
  const parent = new Map();
  const ids = [...graph.adjacency.keys()];
  for (const id of ids) parent.set(id, id);
  const find = (start) => {
    let root = start;
    while (parent.get(root) !== root) root = parent.get(root);
    let node = start;
    while (parent.get(node) !== root) {
      const next = parent.get(node);
      parent.set(node, root);
      node = next;
    }
    return root;
  };
  for (const id of ids) {
    for (const edge of graph.adjacency.get(id)) {
      const left = find(id);
      const right = find(edge.to);
      if (left !== right) parent.set(Math.max(left, right), Math.min(left, right));
    }
  }
  return new Map(ids.map((id) => [id, find(id)]));
}

/** Dense power iteration, deliberately separate from the message-passing implementation. */
function pageRankOracle(graph, damping, iterations = 200) {
  const ids = [...graph.adjacency.keys()].sort((a, b) => a - b);
  const index = new Map(ids.map((id, i) => [id, i]));
  let rank = ids.map(() => 1 / ids.length);
  for (let step = 0; step < iterations; step++) {
    const next = ids.map(() => 0);
    let dangling = 0;
    for (const id of ids) {
      const outDegree = graph.adjacency.get(id).length;
      if (outDegree === 0) {
        dangling += rank[index.get(id)] / ids.length;
        continue;
      }
      const share = rank[index.get(id)] / outDegree;
      for (const edge of graph.adjacency.get(id)) next[index.get(edge.to)] += damping * share;
    }
    rank = next.map((value) => value + (1 - damping) / ids.length + damping * dangling);
  }
  const total = rank.reduce((acc, value) => acc + value, 0);
  return new Map(ids.map((id, i) => [id, rank[i] / total]));
}

module.exports = { MASK64, INFINITY, SplitMix64, digest, buildGraph, assignPartitions, REDUCERS,
  VertexState, PregelEngine, ssspProgram, componentsProgram, pageRankProgram, superstepProbeProgram,
  dijkstra, unionFindComponents, pageRankOracle };

if (require.main === module) {
  const assert = require("node:assert");
  const ssspInitial = (graph, source) => (id) => ssspProgram.initial(id, graph, source);

  // --- SSSP against Dijkstra -------------------------------------------------
  const graph = buildGraph({ vertices: 26, extraEdges: 24, seed: 843 });
  const source = 0;
  const ssspEngine = new PregelEngine(graph, { partitions: 4, checkpointEvery: 3,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators });
  ssspEngine.reset(ssspInitial(graph, source));
  const ssspRun = ssspEngine.run(ssspProgram, { initialValue: ssspInitial(graph, source) });
  const oracleDistances = dijkstra(graph, source);
  for (const id of graph.adjacency.keys()) {
    assert.strictEqual(ssspEngine.valueOf(id), oracleDistances.get(id), `SSSP distance to ${id}`);
  }
  assert.strictEqual(ssspEngine.valueOf(source), 0, "the source is at distance 0");
  assert.strictEqual(ssspRun.halted, true, "SSSP reaches a quiescent state");
  assert.ok(ssspRun.supersteps > 1 && ssspRun.supersteps <= graph.vertices + 2,
    `SSSP halts in ${ssspRun.supersteps} supersteps (diameter-bounded)`);
  const reachable = [...oracleDistances.values()].filter((value) => value !== INFINITY).length;
  assert.ok(reachable >= graph.vertices / 2, `most vertices are reachable (${reachable})`);

  // --- connected components against union-find -------------------------------
  const undirected = buildGraph({ vertices: 30, extraEdges: 18, seed: 844, directed: false,
    weighted: false, connected: false });
  assert.ok(new Set(unionFindComponents(undirected).values()).size >= 3,
    "the component graph really is disconnected");
  const ccEngine = new PregelEngine(undirected, { partitions: 5, checkpointEvery: 2,
    combiner: componentsProgram.combiner, aggregators: componentsProgram.aggregators });
  ccEngine.reset((id) => componentsProgram.initial(id));
  const ccRun = ccEngine.run(componentsProgram, { initialValue: (id) => componentsProgram.initial(id) });
  const oracleComponents = unionFindComponents(undirected);
  for (const id of undirected.adjacency.keys()) {
    assert.strictEqual(ccEngine.valueOf(id), oracleComponents.get(id), `component label of ${id}`);
  }
  const distinctLabels = new Set([...ccEngine.states.values()].map((state) => state.value));
  assert.strictEqual(distinctLabels.size, new Set(oracleComponents.values()).size,
    "the component count matches the oracle");
  assert.ok(distinctLabels.size >= 3, `several components were found (${distinctLabels.size})`);
  assert.ok(ccRun.supersteps <= Math.ceil(Math.log2(undirected.vertices)) + 4,
    `label propagation converged in ${ccRun.supersteps} supersteps`);

  // --- PageRank against dense power iteration --------------------------------
  const web = buildGraph({ vertices: 20, extraEdges: 12, seed: 845, weighted: false });
  const damping = 0.85;
  const prProgram = pageRankProgram(damping, web.vertices);
  const prEngine = new PregelEngine(web, { partitions: 4, checkpointEvery: 4,
    combiner: prProgram.combiner, aggregators: prProgram.aggregators });
  prEngine.reset(() => 1 / web.vertices);
  prEngine.run(prProgram, { initialValue: () => 1 / web.vertices, maxSupersteps: 120 });
  const prOracle = pageRankOracle(web, damping);
  let maxRankError = 0;
  let rankSum = 0;
  for (const id of web.adjacency.keys()) {
    maxRankError = Math.max(maxRankError, Math.abs(prEngine.valueOf(id) - prOracle.get(id)));
    rankSum += prEngine.valueOf(id);
  }
  assert.ok(maxRankError < 1e-5,
    `PageRank matches power iteration (max error ${maxRankError.toExponential(2)})`);
  assert.ok(Math.abs(rankSum - 1) < 1e-9, `ranks sum to 1 (${rankSum})`);
  assert.strictEqual(prEngine.aggregateLog.length, 120, "the iterative run used its full budget");
  assert.ok(prEngine.aggregateLog.every((entry) => Math.abs(entry.totalRank - 1) < 1e-6),
    "the total rank aggregator is 1 at every barrier");

  // --- superstep barrier safety ---------------------------------------------
  const probeEngine = new PregelEngine(graph, { partitions: 3, checkpointEvery: 100 });
  probeEngine.reset(() => superstepProbeProgram.initial());
  probeEngine.run(superstepProbeProgram, { initialValue: () => 0, maxSupersteps: 30 });
  for (const id of graph.adjacency.keys()) {
    assert.strictEqual(probeEngine.states.get(id).barrierViolation ?? false, false,
      `no same-superstep delivery at ${id}`);
  }
  const probeValues = [...graph.adjacency.keys()].filter((id) => graph.adjacency.get(id).length > 0);
  assert.ok(probeValues.length > 10, "the probe touched many vertices");
  for (const id of probeValues) {
    const inDegree = [...graph.adjacency.values()]
      .reduce((acc, edges) => acc + edges.filter((edge) => edge.to === id).length, 0);
    assert.strictEqual(probeEngine.valueOf(id), 1 + inDegree, `probe tick count at ${id}`);
  }

  // --- combiners cut traffic without changing results ------------------------
  const plain = new PregelEngine(graph, { partitions: 4, checkpointEvery: 100,
    aggregators: ssspProgram.aggregators });
  plain.reset(ssspInitial(graph, source));
  const plainRun = plain.run(ssspProgram, { initialValue: ssspInitial(graph, source),
    maxSupersteps: 200 });
  assert.strictEqual(plain.stateDigest(), ssspEngine.stateDigest(),
    "the combiner does not change the answer");
  assert.strictEqual(plainRun.supersteps, ssspRun.supersteps, "nor the schedule");
  assert.ok(ssspEngine.messagesDelivered <= plain.messagesDelivered,
    `combining reduced delivered traffic (${ssspEngine.messagesDelivered} <= ${plain.messagesDelivered})`);
  assert.ok(ssspEngine.messagesCombined > 0,
    `the combiner merged ${ssspEngine.messagesCombined} messages`);
  assert.strictEqual(ssspEngine.messagesSent,
    ssspEngine.messagesDelivered + ssspEngine.messagesCombined,
  "every emitted message is either delivered or merged away");
  assert.strictEqual(ssspEngine.messagesSent,
    ssspEngine.localMessages + ssspEngine.crossPartitionMessages,
  "every message is either local or crosses a partition");
  assert.ok(ssspEngine.crossPartitionMessages > 0, "some traffic really does cross partitions");

  // --- fault tolerance: coordinated rollback is exact ------------------------
  const faulty = new PregelEngine(graph, { partitions: 4, checkpointEvery: 3,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators });
  faulty.reset(ssspInitial(graph, source));
  faulty.run(ssspProgram, { initialValue: ssspInitial(graph, source),
    faults: [{ superstep: 4, worker: 1 }, { superstep: 7, worker: 3 }] });
  assert.strictEqual(faulty.faultEvents.length, 2, "both injected faults fired");
  assert.strictEqual(faulty.stateDigest(), ssspEngine.stateDigest(),
    "the recovered run reaches exactly the failure-free state");
  for (const id of graph.adjacency.keys()) {
    assert.strictEqual(faulty.valueOf(id), oracleDistances.get(id), `recovered distance to ${id}`);
  }
  assert.ok(faulty.replayedSupersteps > 0,
    `rollback replayed ${faulty.replayedSupersteps} supersteps`);
  assert.ok(faulty.replayedSupersteps <= 2 * faulty.checkpointEvery,
    "no rollback exceeds the checkpoint interval");
  assert.ok(faulty.messagesReplayed >= faulty.replayedSupersteps,
    `replayed supersteps re-sent ${faulty.messagesReplayed} messages`);
  assert.ok(faulty.checkpoints.length >= 3, "progress was checkpointed");
  for (const snapshot of faulty.checkpoints) {
    assert.strictEqual(snapshot.states.length, graph.vertices, "a checkpoint covers every vertex");
    assert.strictEqual(snapshot.digest.length, 16, "a checkpoint carries a digest");
  }
  const restored = new PregelEngine(graph, { partitions: 4 });
  restored.reset(ssspInitial(graph, source));
  restored.restore(faulty.checkpoints[1]);
  assert.strictEqual(restored.states.size, faulty.checkpoints[1].states.length,
    "restoring reproduces the vertex count");
  assert.strictEqual(digest([...restored.states.values()]
    .map((state) => [state.id, state.value, state.halted, state.inbox.length])),
  faulty.checkpoints[1].digest, "and reproduces the checkpoint digest exactly");

  // --- the negative control: skipping recovery corrupts the answer -----------
  const broken = new PregelEngine(graph, { partitions: 4, checkpointEvery: 3,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators, recovery: "none" });
  broken.reset(ssspInitial(graph, source));
  broken.run(ssspProgram, { initialValue: ssspInitial(graph, source),
    faults: [{ superstep: 4, worker: 1 }] });
  assert.notStrictEqual(broken.stateDigest(), ssspEngine.stateDigest(),
    "an engine that skips recovery really does produce a different answer");

  // --- boundary graphs -------------------------------------------------------
  const single = { adjacency: new Map([[7, []]]), vertices: 1, directed: true };
  const singleEngine = new PregelEngine(single, { partitions: 1, checkpointEvery: 1,
    aggregators: ssspProgram.aggregators });
  singleEngine.reset(() => 0);
  singleEngine.run(ssspProgram, { initialValue: () => 0 });
  assert.strictEqual(singleEngine.valueOf(7), 0, "a single vertex already holds the answer");
  assert.deepStrictEqual(singleEngine.messagesSent, 0, "and sends nothing");

  const empty = { adjacency: new Map(), vertices: 0, directed: true };
  const emptyEngine = new PregelEngine(empty, { partitions: 2, aggregators: ssspProgram.aggregators });
  emptyEngine.reset(() => 0);
  emptyEngine.run(ssspProgram, { initialValue: () => 0 });
  assert.strictEqual(emptyEngine.finished(), true, "an empty graph finishes immediately");

  const isolated = { adjacency: new Map([[0, [{ to: 1, weight: 5 }]], [1, []], [2, []], [3, []]]),
    vertices: 4, directed: true };
  const isolatedEngine = new PregelEngine(isolated, { partitions: 2, checkpointEvery: 1,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators });
  isolatedEngine.reset(ssspInitial(isolated, 0));
  isolatedEngine.run(ssspProgram, { initialValue: ssspInitial(isolated, 0) });
  assert.deepStrictEqual([isolatedEngine.valueOf(0), isolatedEngine.valueOf(1),
    isolatedEngine.valueOf(2), isolatedEngine.valueOf(3)], [0, 5, INFINITY, INFINITY],
  "unreachable vertices stay at infinity");

  const selfLoop = { adjacency: new Map([[0, [{ to: 0, weight: 3 }]], [1, [{ to: 0, weight: 1 }]]]),
    vertices: 2, directed: true };
  const selfLoopEngine = new PregelEngine(selfLoop, { partitions: 2,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators });
  selfLoopEngine.reset(ssspInitial(selfLoop, 0));
  selfLoopEngine.run(ssspProgram, { initialValue: ssspInitial(selfLoop, 0) });
  assert.strictEqual(selfLoopEngine.valueOf(0), 0, "a self loop cannot improve the source");
  assert.strictEqual(selfLoopEngine.valueOf(1), INFINITY, "and the sink stays unreachable");

  const twoNodes = { adjacency: new Map([[0, [{ to: 1, weight: 4 }]], [1, [{ to: 0, weight: 4 }]]]),
    vertices: 2, directed: true };
  const twoNodeEngine = new PregelEngine(twoNodes, { partitions: 2,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators });
  twoNodeEngine.reset(ssspInitial(twoNodes, 1));
  twoNodeEngine.run(ssspProgram, { initialValue: ssspInitial(twoNodes, 1) });
  assert.deepStrictEqual([twoNodeEngine.valueOf(0), twoNodeEngine.valueOf(1)], [4, 0],
    "a two-node cycle is solved from the non-zero source");

  // --- determinism and partitioning -----------------------------------------
  const replayEngine = new PregelEngine(graph, { partitions: 4, checkpointEvery: 3,
    combiner: ssspProgram.combiner, aggregators: ssspProgram.aggregators });
  replayEngine.reset(ssspInitial(graph, source));
  replayEngine.run(ssspProgram, { initialValue: ssspInitial(graph, source) });
  assert.strictEqual(replayEngine.stateDigest(), ssspEngine.stateDigest(), "runs are reproducible");
  assert.deepStrictEqual(replayEngine.aggregateLog, ssspEngine.aggregateLog,
    "the aggregator history is reproducible");
  const partitionLoad = new Map();
  for (const id of graph.adjacency.keys()) {
    const worker = ssspEngine.partitionOf.get(id);
    partitionLoad.set(worker, (partitionLoad.get(worker) ?? 0) + 1);
  }
  const loads = [...partitionLoad.values()];
  assert.strictEqual(loads.length, 4, "four workers were used");
  assert.ok(Math.max(...loads) - Math.min(...loads) <= 1, `partitions are balanced (${loads.join(",")})`);
  assert.ok(ssspEngine.messagesSent > graph.vertices, `messages sent: ${ssspEngine.messagesSent}`);

  console.log(`sssp: vertices=${graph.vertices} supersteps=${ssspRun.supersteps} ` +
    `sent=${ssspEngine.messagesSent} delivered=${ssspEngine.messagesDelivered} ` +
    `cross-partition=${ssspEngine.crossPartitionMessages}`);
  console.log(`components: supersteps=${ccRun.supersteps} labels=${distinctLabels.size}; ` +
    `pagerank: max-error=${maxRankError.toExponential(2)} sum=${rankSum.toFixed(12)}`);
  console.log(`faults=${faulty.faultEvents.length} replayed-supersteps=${faulty.replayedSupersteps} ` +
    `replayed-messages=${faulty.messagesReplayed} checkpoints=${faulty.checkpoints.length}`);
  console.log("PASS");
}
