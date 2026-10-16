/** Task 939 - Implement trace-based replay system for production debugging.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A request is executed over a fixed service graph (gateway -> auth/orders -> payments -> db) and
 * every span is recorded with its host's *skewed* clock: the timestamps of a span have an unknown
 * per-host offset added, while durations stay skew-free because they are measured on one host. The
 * replay engine rebuilds the span tree, recovers the per-host offsets from the containment
 * constraints a parent/child pair must satisfy (child inside parent) by relaxing them to a fixpoint
 * like Bellman-Ford, then re-executes every operator from its recorded input and compares the result
 * to the recorded output - reporting the divergent spans when a service is not deterministic.
 *
 * Simplified: the graph is fixed, one span per service per request, offsets are constant per host
 * and a single global shift of the corrected trace is unidentifiable (reported, not fixed). Verified:
 * containment before/after correction, the offset estimate against the simulator's ground truth,
 * self-time accounting, the critical path against the true one, depth/empty/single-span boundaries,
 * injected nondeterminism detection, and determinism of the whole pipeline.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}
const hashName = (name) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i += 1) h = Math.imul(h ^ name.charCodeAt(i), 0x01000193) >>> 0;
  return h >>> 0;
};

const SKEW = 60;
/** Ground truth: the constant clock offset a host adds to the timestamps it records. */
const trueOffset = (service) => (mix32(hashName(service)) % (2 * SKEW + 1)) - SKEW;

const EDGES = { gateway: ["auth", "orders"], auth: ["db"], orders: ["db", "payments"], payments: ["db"], db: [] };
const FALLBACK = (x) => (x * 13 + 5) % 1000;
const FUNCS = { gateway: (x) => (x * 2 + 1) % 1000, auth: (x) => (x * 3 + 7) % 1000, orders: (x) => (x * 5 + 3) % 1000,
  payments: (x) => (x * 7 + 11) % 1000, db: (x) => (x * 11 + 13) % 1000 };
const compute = (service, input) => (FUNCS[service] ?? FALLBACK)(input);

/** Simulate one request: `recorded` carries skewed timestamps, `truth` the real clock. */
function simulateTrace({ seed, root = "gateway", shape = (service) => EDGES[service] ?? [] }) {
  const rng = splitmix64(seed);
  const truth = [];
  let nextId = 1;
  const visit = (service, parent, startTrue, depth) => {
    const own = 1 + rng.int(5);
    const input = rng.int(1000);
    const span = { id: nextId++, parent, service, depth, input, output: compute(service, input), trueStart: startTrue };
    truth.push(span);
    let end = startTrue + own;
    for (const child of shape(service, depth)) end = Math.max(end, visit(child, span.id, end, depth + 1));
    span.duration = end - startTrue;
    span.trueEnd = end;
    return end;
  };
  visit(root, null, 0, 0);
  const recorded = truth.map((s) => ({ id: s.id, parent: s.parent, service: s.service, input: s.input, output: s.output,
    start: s.trueStart + trueOffset(s.service), duration: s.duration }));
  return { recorded, truth };
}

const index = (spans) => new Map(spans.map((s) => [s.id, s]));
function childrenOf(spans) {
  const children = new Map();
  for (const s of spans) if (s.parent !== null) children.set(s.parent, [...(children.get(s.parent) ?? []), s.id]);
  return children;
}

/**
 * Recover the per-host clock offsets. Every parent/child pair yields
 * lo <= o[child] - o[parent] <= hi from the containment constraints (a child starts no earlier than
 * its parent and ends no later). The two difference systems are folded into one graph of upper
 * bounds, and Bellman-Ford shortest paths from the root host (pinned to zero) give the pointwise
 * largest feasible correction - the timeline repair a trace tool applies. The absolute offset stays
 * unidentifiable, so the estimate differs from the truth by one global shift shared by all hosts.
 */
function estimateOffsets(spans) {
  const root = spans.find((s) => s.parent === null);
  const offsets = new Map();
  if (!root) return offsets;
  const byId = index(spans);
  const services = [...new Set(spans.map((s) => s.service))];
  const dist = new Map(services.map((s) => [s, Infinity]));
  dist.set(root.service, 0);
  const edges = [];
  for (const span of spans) {
    const parent = byId.get(span.parent);
    if (!parent) continue;
    const lo = parent.start - span.start;
    const hi = parent.start + parent.duration - span.start - span.duration;
    edges.push([parent.service, span.service, hi], [span.service, parent.service, -lo]);
  }
  for (let pass = 0; pass <= services.length; pass += 1) {
    let changed = false;
    for (const [from, to, weight] of edges) {
      if (!Number.isFinite(dist.get(from))) continue;
      if (dist.get(from) + weight < dist.get(to)) { dist.set(to, dist.get(from) + weight); changed = true; }
    }
    if (!changed) break;
  }
  for (const service of services) offsets.set(service, Number.isFinite(dist.get(service)) ? dist.get(service) : 0);
  return offsets;
}

/** The chain of spans that ended last at every level - the spans that set the trace's duration. */
function criticalPath(spans, children) {
  const byId = index(spans);
  const path = [];
  let node = spans.find((s) => s.parent === null);
  while (node) {
    path.push({ id: node.id, service: node.service, end: node.end ?? node.trueEnd });
    const long_ = (children.get(node.id) ?? []).map((id) => byId.get(id)).filter((k) => (k.end ?? k.trueEnd) === (node.end ?? node.trueEnd));
    node = long_.length ? long_[0] : null;
  }
  return path;
}

let replayCalls = 0;
/** Replay: corrected timestamps, self times, critical path and a per-span output comparison. */
function replay(trace, { buggy = new Set() } = {}) {
  const children = childrenOf(trace);
  const offsets = estimateOffsets(trace);
  const corrected = trace.map((s) => ({ ...s, start: s.start + offsets.get(s.service),
    end: s.start + offsets.get(s.service) + s.duration }));
  const byId = index(corrected);
  const containment = corrected.every((s) => {
    const parent = byId.get(s.parent);
    return !parent || (s.start >= parent.start && s.end <= parent.end);
  });
  const rawContainment = trace.every((s) => {
    const parent = trace.find((p) => p.id === s.parent);
    return !parent || (s.start >= parent.start && s.start + s.duration <= parent.start + parent.duration);
  });
  const self = corrected.map((s) => ({ ...s, self: s.duration - (children.get(s.id) ?? []).reduce((a, c) => a + byId.get(c).duration, 0) }));
  const divergences = [];
  for (const span of trace) {
    const output = compute(span.service, span.input) + (buggy.has(span.service) ? ++replayCalls : 0);
    if (output !== span.output) divergences.push({ id: span.id, service: span.service, recorded: span.output, replayed: output });
  }
  return { offsets, corrected, self, containment, rawContainment,
    path: criticalPath(corrected, children), divergences, status: divergences.length ? "divergent" : "reproduced" };
}

const totalSelf = (result) => result.self.reduce((a, s) => a + s.self, 0);
const pathNames = (path) => path.map((p) => p.service).join(">");

function main() {
  const { recorded, truth } = simulateTrace({ seed: 939 });

  // 1. Shape and wall time: seven spans, and the corrected root spans the whole request.
  assert(recorded.length === 7 && Math.max(...truth.map((s) => s.depth)) === 3, `trace shape, got ${recorded.length} spans`);
  const result = replay(recorded);
  const rootEnd = Math.max(...result.corrected.map((s) => s.end));
  assert(result.corrected[0].end === rootEnd && result.corrected[0].duration === truth[0].trueEnd, "the root closes the trace after correction");

  // 2. Self-time accounting: self times are non-negative and sum to the request's wall time.
  assert(result.self.every((s) => s.self >= 0), "no span has negative self time");
  assert(totalSelf(result) === result.corrected[0].duration, `self times must sum to the root duration: ${totalSelf(result)}`);

  // 3. Clock skew exists and is removed: containment fails on the raw trace, holds after correction.
  assert(!result.rawContainment, "the raw recorded trace violates parent/child containment");
  assert(result.containment, "every span sits inside its parent after skew correction");

  // 4. Oracle for the estimator: the correction inverts the ground-truth skew up to one global shift.
  const residuals = [...result.offsets].map(([service, estimate]) => estimate + trueOffset(service));
  assert(Math.max(...residuals) - Math.min(...residuals) <= 4, `offsets recovered up to one global shift: ${residuals}`);

  // 5. Critical path: the corrected path equals the one derived from the true clock.
  const truePath = criticalPath(truth, childrenOf(truth));
  assert(pathNames(result.path) === pathNames(truePath), `path ${pathNames(result.path)} vs truth ${pathNames(truePath)}`);
  assert(result.path.at(-1).end === rootEnd && result.path.length >= 3, "the critical path is the long pole of the trace");

  // 6. Boundary: a single-span trace has no constraints, so the only offset is zero.
  const single = simulateTrace({ seed: 1, shape: () => [] });
  const singleResult = replay(single.recorded);
  assert(singleResult.offsets.size === 1 && singleResult.offsets.get("gateway") === 0, "a lone host needs no skew estimate");
  assert(pathNames(singleResult.path) === "gateway" && totalSelf(singleResult) === single.recorded[0].duration, "single-span accounting holds");

  // 7. Boundary: an empty trace replays to an empty result without throwing.
  const empty = replay([]);
  assert(empty.corrected.length === 0 && empty.path.length === 0 && empty.containment && empty.status === "reproduced", "empty trace is a no-op");

  // 8. Boundary: a deep chain still terminates and stays consistent.
  const deep = simulateTrace({ seed: 8, root: "layer0", shape: (service, depth) => (depth < 39 ? [`layer${depth + 1}`] : []) });
  const deepResult = replay(deep.recorded);
  assert(deep.recorded.length === 40 && deepResult.containment, "a 40-deep chain is corrected");
  assert(totalSelf(deepResult) === deepResult.corrected[0].duration, "deep chain self times still sum correctly");

  // 9. Nondeterminism: replay reproduces a deterministic trace and pinpoints an injected fault.
  assert(result.status === "reproduced" && result.divergences.length === 0, "a deterministic service reproduces its outputs");
  const faulty = replay(simulateTrace({ seed: 939 }).recorded, { buggy: new Set(["payments"]) });
  assert(faulty.status === "divergent" && faulty.divergences.length === 1, `exactly the faulty span diverges: ${faulty.divergences.length}`);
  assert(faulty.divergences[0].service === "payments", "the divergence is attributed to the injected service");
  assert(faulty.divergences[0].recorded !== faulty.divergences[0].replayed, "divergence records both values");

  // 10. Determinism: the same seed records and replays identically, another seed does not.
  assert(JSON.stringify(simulateTrace({ seed: 939 }).recorded) === JSON.stringify(recorded), "recording is deterministic");
  assert(JSON.stringify(replay(simulateTrace({ seed: 940 }).recorded).path) !== JSON.stringify(result.path), "another seed yields another trace");
  assert(JSON.stringify(replay(recorded).self) === JSON.stringify(result.self), "replay is deterministic");

  console.log(`trace-replay: spans=${recorded.length} depth=3 offsets=${[...result.offsets.values()].join("/")} divergences=${faulty.divergences.length}`);
  console.log("PASS");
}
module.exports = { splitmix64, simulateTrace, replay, estimateOffsets, criticalPath, childrenOf, trueOffset, SKEW };
if (require.main === module) main();
