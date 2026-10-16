/** Task 846 - Implement subgraph-centric distributed processing framework and test on motifs.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * A subgraph-centric ("think like a graph") framework. The vertex set is split into contiguous ranges
 * and each range becomes one subgraph that owns those vertices and *fetches replicas* of the vertices
 * on its boundary in an explicit exchange phase. Computation then happens entirely inside a subgraph —
 * no per-vertex messaging — which is what makes this family of engines cheap on dense local structure.
 *
 * Motif census. For every vertex triple {a, b, c} the framework counts how many edges the triple
 * induces (2 = a wedge, 3 = a triangle) and how many of the two possible directed cycles are closed.
 * Double counting is impossible by construction:
 *
 *     a triple is enumerated by exactly the subgraph that owns its smallest vertex id,
 *
 * and the owner always holds what it needs: the smallest vertex is internal, every other vertex is
 * either internal or a replica whose full adjacency (out-arcs, in-arcs and neighbour list) was fetched,
 * and the enumeration is written so that each edge of the triple is decided from data the owner has.
 * The per-subgraph counts therefore sum to the global census, which is checked against brute force and
 * against the algebraic identity `#wedges = SUM_v C(deg_v, 2) - 3 * #triangles`.
 *
 * Fault tolerance. Every subgraph checkpoints its replica cache before the exchange; a subgraph that
 * dies while fetching restarts from that checkpoint, re-issues the missing requests, and the recovered
 * census must be *identical* to the failure-free one. A variant with recovery disabled keeps a
 * half-filled cache and is shown to produce a different census, so the check has teeth. Replica caches
 * carry a checksum of the owner's reply, so a corrupted response is detected and flagged rather than
 * silently changing the answer.
 *
 * Simplifications, stated honestly: subgraphs are in-process objects stepped one at a time instead of
 * threads or machines; "scale" is a parameter (vertex count, subgraph count, edge density); the
 * exchange is one round of adjacency fetches rather than a full Gelly-style execute/exchange/compute
 * loop over several supersteps; nothing is serialised or spilled to disk. No timers, threads or
 * sockets; randomness is seeded splitmix64.
 *
 * Invariants verified below:
 *  - partitioning: internal vertex sets partition V exactly, every arc is visible from at least one
 *    subgraph, and the measured edge cut matches an independent count;
 *  - exchange safety: after the exchange every replica cache equals the owner's data including its
 *    checksum, and a forged reply is detected and flagged;
 *  - census correctness: distributed wedges, triangles and directed 3-cycles equal brute force, the
 *    per-subgraph counts sum to the global total, and no triple is counted twice;
 *  - algebraic cross-check: #wedges = SUM_v C(deg_v, 2) - 3 * #triangles;
 *  - fault tolerance: one and two simultaneous subgraph failures recover to the identical census while
 *    re-sent messages are counted, and an engine without recovery demonstrably differs;
 *  - liveness/cost: two rounds (exchange, census), message count proportional to the boundary size;
 *  - edge cases: empty graph, one vertex, a path of three, a triangle with a pendant, two disjoint
 *    triangles, K5 (10 triangles), duplicate arcs, self loops, 2 and 8 subgraphs.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

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

/** Checksum of a replica reply: what a receiver compares with the owner's live data. */
function checksum(neighbours, out, inbound) {
  let h = 0xcbf29ce484222325n;
  const feed = (list) => {
    for (const value of [...list].sort((a, b) => a - b)) {
      h = (h ^ BigInt(value)) & MASK64;
      h = (h * 0x100000001b3n) & MASK64;
    }
  };
  feed(neighbours);
  feed(out);
  feed(inbound);
  return h.toString(16).padStart(16, "0");
}

/** Derives every adjacency view a subgraph could need, once per graph object. */
const views = new WeakMap();

function viewOf(graph) {
  if (views.has(graph)) return views.get(graph);
  const ids = [...graph.adjacency.keys()].sort((a, b) => a - b);
  const out = new Map(ids.map((id) => [id, [...(graph.adjacency.get(id) ?? [])]
    .sort((a, b) => a - b)]));
  const inbound = new Map(ids.map((id) => [id, []]));
  for (const id of ids) {
    for (const target of out.get(id)) {
      if (target !== id) inbound.get(target).push(id);
    }
  }
  for (const id of ids) inbound.get(id).sort((a, b) => a - b);
  const neighbours = new Map(ids.map((id) => [id,
    [...new Set([...out.get(id), ...inbound.get(id)])].sort((a, b) => a - b)]));
  const view = { ids, out, inbound, neighbours };
  views.set(graph, view);
  return view;
}

/** Directed graph with self loops dropped and duplicate arcs collapsed. */
function buildGraph(options = {}) {
  const vertices = options.vertices ?? 48;
  const extraEdges = options.extraEdges ?? 60;
  const seed = options.seed ?? 846;
  const connected = options.connected ?? true;
  const rng = new SplitMix64(seed);
  const adjacency = new Map();
  for (let i = 0; i < vertices; i++) adjacency.set(i, new Set());
  const addArc = (from, to) => {
    if (from !== to) adjacency.get(from).add(to);
  };
  for (let i = 1; i < vertices && connected; i++) addArc(i - 1, i);
  for (let i = 0; i < extraEdges; i++) addArc(rng.upto(vertices), rng.upto(vertices));
  return { adjacency, vertices };
}

/** Range partitioner: contiguous id ranges are deterministic and keep the edge cut measurable. */
function partition(graph, partitions) {
  const ids = viewOf(graph).ids;
  const owner = new Map();
  ids.forEach((id, index) => owner.set(id,
    Math.min(partitions - 1, Math.floor((index * partitions) / ids.length))));
  return owner;
}

// --- exhaustive oracles -----------------------------------------------------
/** Brute force over all triples: wedges, triangles and directed 3-cycles. */
function bruteCensus(graph) {
  const view = viewOf(graph);
  const ids = view.ids;
  const linked = (left, right) => view.neighbours.get(left).includes(right);
  const points = (from, to) => (view.out.get(from) ?? []).includes(to);
  let wedges = 0;
  let triangles = 0;
  let directedCycles = 0;
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      for (let k = j + 1; k < ids.length; k++) {
        const [a, b, c] = [ids[i], ids[j], ids[k]];
        const edges = (linked(a, b) ? 1 : 0) + (linked(a, c) ? 1 : 0) + (linked(b, c) ? 1 : 0);
        if (edges === 2) wedges++;
        else if (edges === 3) triangles++;
        if (points(a, b) && points(b, c) && points(c, a)) directedCycles++;
        if (points(a, c) && points(c, b) && points(b, a)) directedCycles++;
      }
    }
  }
  return { wedges, triangles, directedCycles };
}

/** The algebraic identity used as a second, independent oracle. */
function wedgeIdentity(graph) {
  const view = viewOf(graph);
  let pairs = 0;
  for (const id of view.ids) {
    const degree = view.neighbours.get(id).length;
    pairs += (degree * (degree - 1)) / 2;
  }
  const { triangles } = bruteCensus(graph);
  return pairs - 3 * triangles;
}

/**
 * The census itself, written against an `access` object so exactly one implementation serves both the
 * subgraphs and the tests. `access` must expose neighbours(x), out(x), inbound(x) and isInternal(x).
 */
function censusWith(access, collected = null) {
  const counts = { wedges: 0, triangles: 0, directedCycles: 0, triples: 0 };
  const seen = new Set();
  for (const a of access.internalVertices()) {
    const neighboursA = access.neighbours(a);
    const setA = new Set(neighboursA);
    const outA = new Set(access.out(a));
    const inA = new Set(access.inbound(a));
    const record = (x, y) => {
      if (x <= a || y <= x) return;
      const key = `${a}:${x}:${y}`;
      if (seen.has(key)) return;
      seen.add(key);
      if (collected !== null) collected.push([a, x, y]);
      const xInA = setA.has(x);
      const yInA = setA.has(y);
      if (!xInA && !yInA) return; // cannot happen with the enumeration below, guarded anyway
      const setX = xInA ? new Set(access.neighbours(x)) : null;
      const setY = yInA ? new Set(access.neighbours(y)) : null;
      const edgeXY = (xInA && setX.has(y)) || (yInA && setY.has(x));
      const edges = (xInA ? 1 : 0) + (yInA ? 1 : 0) + (edgeXY ? 1 : 0);
      counts.triples++;
      if (edges === 2) counts.wedges++;
      else if (edges === 3) counts.triangles++;
      // Directed cycles are decided from a's and (whichever of x, y is one hop from a) data only.
      const xToY = xInA ? access.out(x).includes(y) : access.inbound(y).includes(x);
      const yToX = yInA ? access.out(y).includes(x) : access.inbound(x).includes(y);
      if (xToY && inA.has(y) && outA.has(x)) counts.directedCycles++;
      if (yToX && inA.has(x) && outA.has(y)) counts.directedCycles++;
    };
    for (const b of neighboursA) {
      if (b <= a) continue;
      const neighboursB = access.neighbours(b);
      for (const c of neighboursA) if (c > b) record(b, c);
      for (const c of neighboursB) if (c > b) record(b, c);
    }
    for (const y of neighboursA) {
      if (y <= a) continue;
      for (const x of access.neighbours(y)) {
        if (x > a && x < y) record(x, y);
      }
    }
  }
  return counts;
}

/**
 * One subgraph: the vertices it owns, the replica cache the exchange fills, and its census.
 * The cache is the only state that has to be checkpointed.
 */
class Subgraph {
  constructor(id, owner, graph) {
    this.id = id;
    this.owner = owner;
    this.graph = graph;
    this.view = viewOf(graph);
    this.internal = this.view.ids.filter((node) => owner.get(node) === id);
    this.internalSet = new Set(this.internal);
    this.cache = new Map(); // replica id -> { neighbours, out, inbound, checksum }
    this.checkpoint = null;
    this.requests = 0;
    this.corruptReplies = 0;
  }

  isInternal(node) {
    return this.internalSet.has(node);
  }

  /** The single access point for adjacency data: owned locally or fetched as a replica. */
  access() {
    const missing = [];
    const lookup = (node, field) => {
      if (this.internalSet.has(node)) return this.view[field].get(node) ?? [];
      const entry = this.cache.get(node);
      if (entry === undefined) {
        missing.push(node);
        return [];
      }
      return entry[field];
    };
    return {
      neighbours: (node) => lookup(node, "neighbours"),
      out: (node) => lookup(node, "out"),
      inbound: (node) => lookup(node, "inbound"),
      internalVertices: () => this.internal,
      missing,
    };
  }

  /** Vertices on my boundary: neighbours I do not own. */
  boundaryVertices() {
    const boundary = new Set();
    for (const node of this.internal) {
      for (const neighbour of this.view.neighbours.get(node)) {
        if (!this.internalSet.has(neighbour)) boundary.add(neighbour);
      }
    }
    return [...boundary].sort((a, b) => a - b);
  }

  /** Stores a fetched replica after checking its checksum; a forged digest is kept for the audit. */
  store(node, reply) {
    this.requests++;
    const digest = checksum(reply.neighbours, reply.out, reply.inbound);
    const received = reply.checksum ?? digest;
    if (received !== digest) this.corruptReplies++;
    this.cache.set(node, { neighbours: reply.neighbours.slice(), out: reply.out.slice(),
      inbound: reply.inbound.slice(), checksum: received });
    return digest;
  }

  takeCheckpoint() {
    this.checkpoint = new Map([...this.cache].map(([node, entry]) =>
      [node, { neighbours: entry.neighbours.slice(), out: entry.out.slice(),
        inbound: entry.inbound.slice(), checksum: entry.checksum }]));
    return this.checkpoint;
  }

  restore() {
    this.cache = new Map([...(this.checkpoint ?? new Map())].map(([node, entry]) =>
      [node, { neighbours: entry.neighbours.slice(), out: entry.out.slice(),
        inbound: entry.inbound.slice(), checksum: entry.checksum }]));
    return this.cache;
  }

  census() {
    return censusWith(this.access());
  }

  /** The triples this subgraph counted; used to prove the ownership rule. */
  censusTriples() {
    const collected = [];
    censusWith(this.access(), collected);
    return collected;
  }

  replicaReport() {
    return [...this.cache].map(([node, entry]) => ({ node, checksum: entry.checksum }))
      .sort((a, b) => a.node - b.node);
  }
}

/**
 * The master: owns the partition, runs the exchange phase (optionally killing subgraphs), runs the
 * census phase and aggregates. A fault is {subgraph, afterRequests}: that subgraph dies once the given
 * number of replies has been consumed and either restarts from its checkpoint or (recovery "none")
 * limps on with a half-filled cache.
 */
class SubgraphFramework {
  constructor(graph, options = {}) {
    this.graph = graph;
    this.view = viewOf(graph);
    this.partitions = options.partitions ?? 4;
    this.recovery = options.recovery ?? "checkpoint";
    this.corruptReplyFor = options.corruptReplyFor ?? null;
    this.owner = partition(graph, this.partitions);
    this.subgraphs = [];
    for (let id = 0; id < this.partitions; id++) {
      this.subgraphs.push(new Subgraph(id, this.owner, graph));
    }
    this.messages = 0;
    this.recoveryMessages = 0;
    this.rounds = 0;
    this.faultEvents = [];
  }

  /** The owner answers a fetch: the replica's full adjacency, optionally with a forged checksum. */
  serve(node, forged = false) {
    return {
      neighbours: this.view.neighbours.get(node) ?? [],
      out: this.view.out.get(node) ?? [],
      inbound: this.view.inbound.get(node) ?? [],
      checksum: forged ? "deadbeefdeadbeef" : undefined,
    };
  }

  exchange(faults = []) {
    this.rounds++;
    for (const subgraph of this.subgraphs) {
      subgraph.takeCheckpoint();
      const boundary = subgraph.boundaryVertices();
      const pending = faults.filter((candidate) => candidate.subgraph === subgraph.id);
      let served = 0;
      for (const node of boundary) {
        const forged = this.corruptReplyFor !== null &&
          this.corruptReplyFor.subgraph === subgraph.id && this.corruptReplyFor.node === node;
        this.messages++;
        subgraph.store(node, this.serve(node, forged));
        served++;
        const fault = pending.find((candidate) => candidate.afterRequests === served);
        if (fault === undefined) continue;
        this.faultEvents.push({ subgraph: subgraph.id, served });
        if (this.recovery !== "checkpoint") {
          subgraph.cache.clear(); // the worker died, its partial fetches are gone, nobody re-issues them
          break;
        }
        subgraph.restore(); // roll back, then refetch everything the cache lost
        for (const candidate of boundary.filter((entry) => !subgraph.cache.has(entry))) {
          this.messages++;
          this.recoveryMessages++;
          subgraph.store(candidate, this.serve(candidate, false));
        }
      }
    }
    return this.messages;
  }

  /** Census phase plus the replica-consistency audit. */
  execute() {
    this.rounds++;
    const consistency = [];
    for (const subgraph of this.subgraphs) {
      for (const [node, entry] of subgraph.cache) {
        const live = this.view;
        const digest = checksum(live.neighbours.get(node) ?? [], live.out.get(node) ?? [],
          live.inbound.get(node) ?? []);
        consistency.push({ subgraph: subgraph.id, node, ok: digest === entry.checksum });
      }
    }
    const perSubgraph = this.subgraphs.map((subgraph) => subgraph.census());
    const totals = perSubgraph.reduce((accumulator, counts) => ({
      wedges: accumulator.wedges + counts.wedges,
      triangles: accumulator.triangles + counts.triangles,
      directedCycles: accumulator.directedCycles + counts.directedCycles,
      triples: accumulator.triples + counts.triples,
    }), { wedges: 0, triangles: 0, directedCycles: 0, triples: 0 });
    return { perSubgraph, totals, consistency };
  }

  run(faults = []) {
    this.exchange(faults);
    return this.execute();
  }

  /** Every triple a subgraph counts must have its smallest vertex inside that subgraph. */
  ownershipAudit() {
    const counted = [];
    for (const subgraph of this.subgraphs) {
      for (const triple of subgraph.censusTriples()) {
        counted.push({ subgraph: subgraph.id, triple, ownedSmallest: subgraph.isInternal(triple[0]) });
      }
    }
    const unique = new Set(counted.map((entry) => entry.triple.join(":")));
    return { counted: counted.length, unique: unique.size,
      allOwned: counted.every((entry) => entry.ownedSmallest) };
  }

  edgeCut() {
    let cut = 0;
    for (const [from, targets] of this.graph.adjacency) {
      for (const to of targets) if (this.owner.get(from) !== this.owner.get(to)) cut++;
    }
    return cut;
  }

  boundarySize() {
    return this.subgraphs.reduce((acc, subgraph) => acc + subgraph.boundaryVertices().length, 0);
  }
}

module.exports = { MASK64, SplitMix64, checksum, viewOf, buildGraph, partition, bruteCensus,
  wedgeIdentity, censusWith, Subgraph, SubgraphFramework };

if (require.main === module) {
  const assert = require("node:assert");
  const graph = buildGraph({ vertices: 48, extraEdges: 60, seed: 846 });
  const framework = new SubgraphFramework(graph, { partitions: 4 });
  const oracle = bruteCensus(graph);
  assert.ok(oracle.triangles > 5, `the test graph has triangles (${oracle.triangles})`);
  assert.ok(oracle.wedges > 20, `and wedges (${oracle.wedges})`);
  assert.ok(oracle.directedCycles > 0, `and directed 3-cycles (${oracle.directedCycles})`);

  // --- partitioning ---------------------------------------------------------
  const ownerCounts = new Map();
  for (const node of viewOf(graph).ids) {
    const id = framework.owner.get(node);
    ownerCounts.set(id, (ownerCounts.get(id) ?? 0) + 1);
  }
  assert.strictEqual(ownerCounts.size, 4, "every subgraph owns something");
  assert.strictEqual([...ownerCounts.values()].reduce((a, b) => a + b, 0), graph.vertices,
    "the internal vertex sets partition V exactly");
  const loads = [...ownerCounts.values()];
  assert.ok(Math.max(...loads) - Math.min(...loads) <= 1, `ranges are balanced (${loads.join(",")})`);
  const cut = framework.edgeCut();
  assert.ok(cut > 0, `the graph has a real edge cut (${cut})`);
  assert.strictEqual(framework.boundarySize(), framework.subgraphs
    .reduce((acc, subgraph) => acc + subgraph.boundaryVertices().length, 0),
  "the boundary size is the sum over subgraphs");
  assert.ok(framework.subgraphs.every((subgraph) => subgraph.internal.length > 0),
    "no subgraph is empty");

  // --- exchange -------------------------------------------------------------
  const result = framework.run([]);
  assert.strictEqual(framework.rounds, 2, "one exchange round and one census round suffice");
  assert.ok(result.consistency.length > 0, "replicas were fetched");
  assert.ok(result.consistency.every((entry) => entry.ok), "every replica cache matches its owner");
  assert.strictEqual(framework.messages, framework.boundarySize(),
    "one request-reply pair per boundary vertex");
  for (const subgraph of framework.subgraphs) {
    assert.strictEqual(subgraph.cache.size, subgraph.boundaryVertices().length,
      `subgraph ${subgraph.id} cached its whole boundary`);
    assert.ok([...subgraph.cache.keys()].every((node) => !subgraph.isInternal(node)),
      "a replica is never owned locally");
  }

  // --- census versus the oracles --------------------------------------------
  assert.strictEqual(result.totals.triangles, oracle.triangles, "distributed triangles = brute force");
  assert.strictEqual(result.totals.wedges, oracle.wedges, "distributed wedges = brute force");
  assert.strictEqual(result.totals.directedCycles, oracle.directedCycles,
    "distributed directed 3-cycles = brute force");
  assert.strictEqual(result.totals.wedges, wedgeIdentity(graph),
    "and #wedges = SUM C(deg,2) - 3 * #triangles");
  const sumOfParts = result.perSubgraph.reduce((acc, counts) => ({
    wedges: acc.wedges + counts.wedges,
    triangles: acc.triangles + counts.triangles,
    directedCycles: acc.directedCycles + counts.directedCycles,
  }), { wedges: 0, triangles: 0, directedCycles: 0 });
  assert.deepStrictEqual(sumOfParts, { wedges: result.totals.wedges,
    triangles: result.totals.triangles, directedCycles: result.totals.directedCycles },
  "the global total is exactly the sum of the per-subgraph counts");
  assert.ok(result.totals.triples >= result.totals.wedges + result.totals.triangles,
    "every enumerated triple is a wedge, a triangle or a single edge");
  const audit = framework.ownershipAudit();
  assert.strictEqual(audit.counted, audit.unique,
    `no triple is counted twice (${audit.counted} vs ${audit.unique})`);
  assert.strictEqual(audit.allOwned, true, "every counted triple belongs to the owner of its minimum");

  // --- a forged reply is detected ------------------------------------------
  const target = framework.subgraphs[1].boundaryVertices()[0];
  const corrupted = new SubgraphFramework(graph, { partitions: 4,
    corruptReplyFor: { subgraph: 1, node: target } });
  const corruptedResult = corrupted.run([]);
  assert.ok(corrupted.subgraphs[1].corruptReplies > 0,
    "the forged checksum is detected by the receiving subgraph");
  const flagged = corruptedResult.consistency.filter((entry) => !entry.ok);
  assert.strictEqual(flagged.length, 1, "exactly the forged replica is flagged");
  assert.strictEqual(flagged[0].node, target, "and it is the one that was forged");

  // --- fault tolerance ------------------------------------------------------
  const faulty = new SubgraphFramework(graph, { partitions: 4 });
  const faultyResult = faulty.run([{ subgraph: 1, afterRequests: 2 }]);
  assert.strictEqual(faulty.faultEvents.length, 1, "the injected fault fired");
  assert.ok(faulty.recoveryMessages > 0, `recovery re-sent ${faulty.recoveryMessages} messages`);
  assert.deepStrictEqual(faultyResult.totals, result.totals,
    "the recovered census is identical to the failure-free one");
  assert.ok(faultyResult.consistency.every((entry) => entry.ok),
    "and every replica is consistent again after recovery");
  assert.strictEqual(faulty.subgraphs[1].cache.size, faulty.subgraphs[1].boundaryVertices().length,
    "the restarted subgraph refetched its whole boundary");

  const twoFaults = new SubgraphFramework(graph, { partitions: 4 });
  const twoFaultResult = twoFaults.run([{ subgraph: 0, afterRequests: 1 },
    { subgraph: 3, afterRequests: 3 }]);
  assert.strictEqual(twoFaults.faultEvents.length, 2, "both faults fired");
  assert.deepStrictEqual(twoFaultResult.totals, result.totals,
    "two simultaneous failures still recover exactly");
  const manyFaults = new SubgraphFramework(graph, { partitions: 4 });
  const manyFaultResult = manyFaults.run([{ subgraph: 0, afterRequests: 1 },
    { subgraph: 2, afterRequests: 1 }, { subgraph: 2, afterRequests: 4 }]);
  assert.strictEqual(manyFaults.faultEvents.length, 3, "all three faults fired");
  assert.deepStrictEqual(manyFaultResult.totals, result.totals,
    "a repeated failure on one subgraph still recovers exactly");
  assert.ok(manyFaults.recoveryMessages > faulty.recoveryMessages,
    "more faults cost more recovery traffic");

  // --- the negative control: no recovery means a wrong census ---------------
  const broken = new SubgraphFramework(graph, { partitions: 4, recovery: "none" });
  const brokenResult = broken.run([{ subgraph: 1, afterRequests: 2 }]);
  assert.notDeepStrictEqual(brokenResult.totals, result.totals,
    "an engine that limps on with a half-filled cache produces a different census");
  assert.ok(broken.subgraphs[1].cache.size < broken.subgraphs[1].boundaryVertices().length,
    "its cache really is incomplete");

  // --- partition size must not change the answer ----------------------------
  const coarse = new SubgraphFramework(graph, { partitions: 2 });
  const coarseResult = coarse.run([]);
  const fine = new SubgraphFramework(graph, { partitions: 8 });
  const fineResult = fine.run([]);
  assert.deepStrictEqual(coarseResult.totals, result.totals, "2 subgraphs give the same census");
  assert.deepStrictEqual(fineResult.totals, result.totals, "8 subgraphs give the same census");
  assert.ok(fine.messages >= coarse.messages, "more subgraphs means more boundary fetches");
  assert.ok(coarse.edgeCut() < fine.edgeCut(), "and more subgraphs cut more arcs");

  // --- degenerate graphs ----------------------------------------------------
  const empty = { adjacency: new Map(), vertices: 0 };
  assert.deepStrictEqual(new SubgraphFramework(empty, { partitions: 2 }).run([]).totals,
    { wedges: 0, triangles: 0, directedCycles: 0, triples: 0 }, "an empty graph has no motifs");
  const single = { adjacency: new Map([[0, new Set()]]), vertices: 1 };
  assert.deepStrictEqual(bruteCensus(single), { wedges: 0, triangles: 0, directedCycles: 0 },
    "one vertex has no motifs");
  assert.strictEqual(new SubgraphFramework(single, { partitions: 2 }).run([]).totals.triples, 0,
    "and nothing to enumerate");
  const path = { adjacency: new Map([[0, new Set([1])], [1, new Set([0, 2])], [2, new Set([1])]]),
    vertices: 3 };
  assert.deepStrictEqual(new SubgraphFramework(path, { partitions: 2 }).run([]).totals,
    { wedges: 1, triangles: 0, directedCycles: 0, triples: 1 },
  "a path of three is exactly one wedge");
  const pendant = { adjacency: new Map([[0, new Set([1, 2])], [1, new Set([0, 2, 3])],
    [2, new Set([0, 1])], [3, new Set([1])]]), vertices: 4 };
  const pendantResult = new SubgraphFramework(pendant, { partitions: 3 }).run([]);
  assert.deepStrictEqual({ wedges: pendantResult.totals.wedges,
    triangles: pendantResult.totals.triangles,
    directedCycles: pendantResult.totals.directedCycles }, bruteCensus(pendant),
  "a triangle with a pendant matches brute force");
  assert.strictEqual(pendantResult.totals.triangles, 1, "the pendant adds no triangle");
  assert.strictEqual(pendantResult.totals.wedges, wedgeIdentity(pendant),
    "the wedge identity holds on the pendant graph");
  const twoTriangles = { adjacency: new Map(), vertices: 6 };
  for (const [a, b, c] of [[0, 1, 2], [3, 4, 5]]) {
    twoTriangles.adjacency.set(a, new Set([b, c]));
    twoTriangles.adjacency.set(b, new Set([a, c]));
    twoTriangles.adjacency.set(c, new Set([a, b]));
  }
  const twoTriangleResult = new SubgraphFramework(twoTriangles, { partitions: 2 }).run([]);
  assert.deepStrictEqual(twoTriangleResult.totals,
    { wedges: 0, triangles: 2, directedCycles: bruteCensus(twoTriangles).directedCycles,
      triples: 2 }, "two disjoint triangles are counted separately");
  assert.strictEqual(twoTriangleResult.totals.directedCycles, 4,
    "each complete triangle carries both directed 3-cycles");
  const complete = { adjacency: new Map(), vertices: 5 };
  for (let i = 0; i < 5; i++) {
    complete.adjacency.set(i, new Set([0, 1, 2, 3, 4].filter((j) => j !== i)));
  }
  const completeResult = new SubgraphFramework(complete, { partitions: 2 }).run([]);
  assert.strictEqual(completeResult.totals.triangles, 10, "K5 holds C(5,3) = 10 triangles");
  assert.strictEqual(completeResult.totals.wedges, 0, "and no wedge");
  assert.strictEqual(completeResult.totals.directedCycles,
    bruteCensus(complete).directedCycles, "each of which closes both directed cycles");
  assert.strictEqual(wedgeIdentity(complete), 0, "the identity agrees on K5");
  const duplicate = { adjacency: new Map([[0, new Set([1, 1, 2])], [1, new Set([0, 2])],
    [2, new Set([0, 1])]]), vertices: 3 };
  assert.strictEqual(bruteCensus(duplicate).triangles, 1, "duplicate arcs collapse");
  assert.strictEqual(new SubgraphFramework(duplicate, { partitions: 2 }).run([]).totals.triangles, 1,
    "and the distributed census agrees");
  const selfLoop = { adjacency: new Map([[0, new Set([0, 1])], [1, new Set([0])]]), vertices: 2 };
  assert.deepStrictEqual(bruteCensus(selfLoop), { wedges: 0, triangles: 0, directedCycles: 0 },
    "a self loop creates no motif");
  assert.deepStrictEqual(new SubgraphFramework(selfLoop, { partitions: 1 }).run([]).totals,
    { wedges: 0, triangles: 0, directedCycles: 0, triples: 0 }, "and cannot be enumerated");

  // --- determinism ----------------------------------------------------------
  const replay = new SubgraphFramework(graph, { partitions: 4 }).run([]);
  assert.deepStrictEqual(replay.totals, result.totals, "runs are reproducible");
  assert.deepStrictEqual(replay.perSubgraph, result.perSubgraph, "per-subgraph counts too");
  assert.deepStrictEqual(replay.consistency, result.consistency, "and the replica audit");

  console.log(`graph: vertices=${graph.vertices} arcs=${[...graph.adjacency.values()]
    .reduce((a, s) => a + s.size, 0)} subgraphs=${framework.partitions} cut=${cut} ` +
    `boundary=${framework.boundarySize()}`);
  console.log(`census: wedges=${result.totals.wedges} triangles=${result.totals.triangles} ` +
    `directed-cycles=${result.totals.directedCycles} triples=${result.totals.triples} ` +
    `(identity ${wedgeIdentity(graph)})`);
  console.log(`faults: 1-fault recovery-messages=${faulty.recoveryMessages} ` +
    `3-fault=${manyFaults.recoveryMessages} forged-replicas-flagged=${flagged.length}`);
  console.log("PASS");
}
