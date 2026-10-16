/** Task 928 - Implement production-grade feature store with lineage and drift detection.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 *   registry/lineage   features carry a version, a source and a transform; the lineage is a DAG of
 *                      source -> transform -> feature nodes with cycle rejection, upstream and
 *                      downstream transitive closure, and impact analysis (which features break when
 *                      a source is lost).
 *   point-in-time      every (entity, feature) keeps a time-sorted version list; the as-of lookup
 *                      binary-searches the newest write with timestamp <= asOf so training never
 *                      sees a value that was written after the label - the offline batch path and
 *                      the online single lookup must return identical vectors.
 *   drift monitor      histogram-based population stability index PSI = sum (cur-ref) ln(cur/ref)
 *                      over fixed bins with epsilon smoothing; PSI >= threshold raises an alert.
 * Simplification: in-memory single-process store, numeric features only, no online serving layer and
 * no automatic backfills. Verified: as-of lookups against a brute-force scan, offline/online
 * consistency, lineage closure against BFS and impact analysis, PSI properties (zero on identical
 * distributions, monotone in the magnitude of a shift, alert thresholds), boundaries, invalid input
 * and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  int(bound) { return this.next() % bound; }
  normal(scale) { return (this.float() + this.float() + this.float() - 1.5) * 2 * scale; }
}

class FeatureRegistry {
  constructor() {
    this.features = new Map();  // name -> {version, source, transform, dtype}
    this.edges = new Map();     // node -> Set of downstream nodes
  }
  define(name, definition) {
    assert.ok(typeof name === "string" && name.length > 0, "a feature name is required");
    assert.ok(definition && typeof definition.source === "string", "a source is required");
    const previous = this.features.get(name);
    const version = previous ? previous.version + 1 : 1;
    this.features.set(name, { version, source: definition.source,
      transform: definition.transform ?? "identity", dtype: definition.dtype ?? "float64" });
    this.addEdge(definition.source, name);
    return version;
  }
  addEdge(from, to) {
    assert.ok(typeof from === "string" && typeof to === "string", "nodes must be named");
    if (!this.edges.has(from)) this.edges.set(from, new Set());
    this.edges.get(from).add(to);
    assert.ok(this.topologicalOrder(), `edge ${from} -> ${to} would create a cycle`);
    return this;
  }
  nodes() {
    const names = new Set(this.edges.keys());
    for (const targets of this.edges.values()) for (const node of targets) names.add(node);
    return [...names];
  }
  /** Kahn's algorithm; returns the order, or null when the graph has a cycle. */
  topologicalOrder() {
    const nodes = this.nodes();
    const indegree = new Map(nodes.map((node) => [node, 0]));
    for (const targets of this.edges.values()) {
      for (const node of targets) indegree.set(node, (indegree.get(node) ?? 0) + 1);
    }
    const queue = nodes.filter((node) => indegree.get(node) === 0);
    const order = [];
    while (queue.length > 0) {
      const node = queue.shift();
      order.push(node);
      for (const next of this.edges.get(node) ?? []) {
        indegree.set(next, indegree.get(next) - 1);
        if (indegree.get(next) === 0) queue.push(next);
      }
    }
    return order.length === nodes.length ? order : null;
  }
  /** Transitive closure via BFS; `upstream` walks the reversed edges. */
  closure(start, direction) {
    assert.ok(this.nodes().includes(start), `unknown lineage node ${start}`);
    const seen = new Set();
    const queue = [start];
    while (queue.length > 0) {
      const node = queue.shift();
      const neighbours = direction === "down" ? [...(this.edges.get(node) ?? [])]
        : [...this.edges.entries()].filter(([, targets]) => targets.has(node)).map(([from]) => from);
      for (const next of neighbours) {
        if (seen.has(next)) continue;
        seen.add(next);
        queue.push(next);
      }
    }
    return seen;
  }
  /** Impact analysis: which downstream features depend, directly or not, on a lost source. */
  impactOf(source) {
    const downstream = this.closure(source, "down");
    return [...downstream].filter((node) => this.features.has(node)).sort();
  }
}

class FeatureStore {
  constructor(registry) {
    this.registry = registry;
    this.values = new Map();   // `${entity}:${feature}` -> [{timestamp, value}]
    this.writes = 0;
  }
  key(entity, feature) { return `${entity}:${feature}`; }
  write(entity, feature, timestamp, value) {
    assert.ok(this.registry.features.has(feature), `unknown feature ${feature}`);
    assert.ok(Number.isInteger(timestamp) && timestamp >= 0, "timestamps are non-negative integers");
    assert.ok(Number.isFinite(value), "feature values must be finite");
    const key = this.key(entity, feature);
    const history = this.values.get(key) ?? [];
    assert.ok(history.length === 0 || history[history.length - 1].timestamp <= timestamp,
      "writes must arrive in non-decreasing timestamp order");
    history.push({ timestamp, value });
    this.values.set(key, history);
    this.writes++;
    return this;
  }
  /** Newest write with timestamp <= asOf, found by binary search. */
  asOf(entity, feature, asOf) {
    assert.ok(this.registry.features.has(feature), `unknown feature ${feature}`);
    const history = this.values.get(this.key(entity, feature));
    if (!history) return null;
    let low = 0, high = history.length - 1, found = -1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (history[mid].timestamp <= asOf) { found = mid; low = mid + 1; } else high = mid - 1;
    }
    return found < 0 ? null : history[found];
  }
  /** Independent oracle: linear scan for the same predicate. */
  bruteForceAsOf(entity, feature, asOf) {
    const history = this.values.get(this.key(entity, feature));
    if (!history) return null;
    let best = null;
    for (const version of history) {
      if (version.timestamp <= asOf && (!best || version.timestamp > best.timestamp)) best = version;
    }
    return best;
  }
  /** Offline training join; returns one vector per request, all from the same as-of semantics. */
  batchAsOf(requests, featureList) {
    return requests.map((request) => {
      const row = { entity: request.entity, asOf: request.asOf, features: {} };
      for (const feature of featureList) {
        const version = this.asOf(request.entity, feature, request.asOf);
        row.features[feature] = version ? version.value : null;
      }
      return row;
    });
  }
  /** Online path: the same lookup, served one request at a time. */
  onlineAsOf(entity, asOf, featureList) {
    const features = {};
    for (const feature of featureList) {
      const version = this.asOf(entity, feature, asOf);
      features[feature] = version ? version.value : null;
    }
    return { entity, asOf, features };
  }
}

function selectBins(values, binCount) {
  const sorted = [...values].sort((a, b) => a - b);
  const edges = [];
  for (let k = 1; k < binCount; k++) edges.push(sorted[Math.floor((k * sorted.length) / binCount)]);
  return edges;
}

function histogram(values, edges) {
  const counts = new Array(edges.length + 1).fill(0);
  for (const value of values) {
    let bin = 0;
    while (bin < edges.length && value > edges[bin]) bin++;
    counts[bin]++;
  }
  return counts.map((count) => count / Math.max(1, values.length));
}

/** Population stability index with epsilon smoothing for empty bins. */
function psi(reference, current, edges, epsilon = 1e-4) {
  assert.strictEqual(reference.length > 0, true, "the reference window must not be empty");
  assert.strictEqual(current.length > 0, true, "the current window must not be empty");
  const left = histogram(reference, edges).map((share) => Math.max(share, epsilon));
  const right = histogram(current, edges).map((share) => Math.max(share, epsilon));
  return left.reduce((acc, share, k) => acc + (right[k] - share) * Math.log(right[k] / share), 0);
}

class DriftMonitor {
  constructor(options = {}) {
    this.threshold = options.threshold ?? 0.2;
    this.binCount = options.binCount ?? 8;
    this.alerts = [];
  }
  check(feature, referenceWindow, currentWindow) {
    const edges = selectBins(referenceWindow, this.binCount);
    const score = psi(referenceWindow, currentWindow, edges);
    const alert = score >= this.threshold;
    if (alert) this.alerts.push({ feature, score });
    return { feature, psi: score, alert, threshold: this.threshold };
  }
}

function main() {
  const prng = new Prng(20250928);

  // 1. Point-in-time correctness: the as-of lookup matches a brute-force scan on random queries.
  const registry = new FeatureRegistry();
  registry.define("clicks_7d", { source: "events_raw", transform: "window_count" });
  registry.define("spend_30d", { source: "orders_raw", transform: "window_sum" });
  const store = new FeatureStore(registry);
  for (let entity = 0; entity < 6; entity++) {
    for (let k = 0; k < 20; k++) {
      store.write(`e${entity}`, "clicks_7d", 5 + k * 5, 10 + entity + k);
      store.write(`e${entity}`, "spend_30d", 5 + k * 5, 100 + entity - k);
    }
  }
  let comparisons = 0;
  for (let trial = 0; trial < 200; trial++) {
    const entity = `e${prng.int(6)}`;
    const feature = prng.float() < 0.5 ? "clicks_7d" : "spend_30d";
    const asOf = prng.int(130);
    const fast = store.asOf(entity, feature, asOf);
    const slow = store.bruteForceAsOf(entity, feature, asOf);
    assert.deepStrictEqual(fast, slow, `as-of mismatch for ${entity}/${feature}@${asOf}`);
    if (fast) assert.ok(fast.timestamp <= asOf, "a future value leaked into the lookup");
    comparisons++;
  }
  assert.ok(comparisons === 200);

  // 2. No label leakage: a value written after the label instant is never returned.
  const leakage = new FeatureStore(registry);
  leakage.write("u1", "clicks_7d", 100, 5);
  leakage.write("u1", "clicks_7d", 200, 99);
  assert.strictEqual(leakage.asOf("u1", "clicks_7d", 150).value, 5);
  assert.strictEqual(leakage.asOf("u1", "clicks_7d", 200).value, 99);
  assert.strictEqual(leakage.asOf("u1", "clicks_7d", 99), null, "before the first write there is nothing");

  // 3. Offline and online paths agree exactly (training/serving consistency).
  const requests = Array.from({ length: 25 }, () => ({ entity: `e${prng.int(6)}`, asOf: prng.int(120) }));
  requests.push({ entity: "e1", asOf: 0 }, { entity: "e2", asOf: 4 }); // predate every write
  const offline = store.batchAsOf(requests, ["clicks_7d", "spend_30d"]);
  for (const row of offline) {
    const online = store.onlineAsOf(row.entity, row.asOf, ["clicks_7d", "spend_30d"]);
    assert.deepStrictEqual(online.features, row.features, "online and offline features differ");
  }
  assert.strictEqual(offline.length, 27);
  assert.ok(offline.some((row) => row.features.clicks_7d === null), "some rows should predate a write");

  // 4. Lineage: transitive closure matches an independent BFS, and impact analysis finds features.
  registry.define("derived_score", { source: "clicks_7d", transform: "normalize" });
  registry.define("risk", { source: "derived_score", transform: "logit" });
  const downstream = registry.closure("events_raw", "down");
  const walked = [];
  { // independent breadth-first walk written against the raw edge map
    const seen = new Set();
    let frontier = ["events_raw"];
    while (frontier.length > 0) {
      const next = [];
      for (const node of frontier) {
        for (const [from, targets] of registry.edges) {
          if (from !== node) continue;
          for (const target of targets) if (!seen.has(target)) { seen.add(target); next.push(target); }
        }
      }
      frontier = next;
    }
    walked.push(...seen);
  }
  assert.deepStrictEqual([...downstream].sort(), walked.sort(), "closure disagrees with the walk");
  assert.deepStrictEqual(registry.impactOf("events_raw"), ["clicks_7d", "derived_score", "risk"]);
  assert.deepStrictEqual(registry.impactOf("orders_raw"), ["spend_30d"]);
  assert.deepStrictEqual([...registry.closure("risk", "up")].sort(), ["clicks_7d", "derived_score", "events_raw"]);
  assert.ok(registry.topologicalOrder().indexOf("events_raw") < registry.topologicalOrder().indexOf("risk"));

  // 5. Drift: PSI is zero on identical windows, grows with the shift, and drives the alert.
  const reference = [];
  const current = [];
  for (let k = 0; k < 500; k++) {
    reference.push(prng.normal(1));
    current.push(prng.normal(1) + 3); // a systematic shift
  }
  const stable = [];
  for (let k = 0; k < 500; k++) stable.push(prng.normal(1));
  const edges = selectBins(reference, 8);
  assert.ok(Math.abs(psi(reference, [...reference], edges)) < 1e-12, "identical windows must score 0");
  const smallShift = reference.map((v) => v + 0.1);
  const mediumShift = reference.map((v) => v + 0.5);
  assert.ok(psi(reference, smallShift, edges) < psi(reference, mediumShift, edges), "PSI must grow with the shift");
  assert.ok(psi(reference, current, edges) > psi(reference, mediumShift, edges));
  const monitor = new DriftMonitor({ threshold: 0.2, binCount: 8 });
  const calm = monitor.check("clicks_7d", reference, stable);
  assert.strictEqual(calm.alert, false, `a stable window alerted with PSI ${calm.psi}`);
  const alarmed = monitor.check("clicks_7d", reference, current);
  assert.strictEqual(alarmed.alert, true, `a shifted window did not alert with PSI ${alarmed.psi}`);
  assert.strictEqual(monitor.alerts.length, 1);
  assert.strictEqual(monitor.alerts[0].feature, "clicks_7d");

  // 6. Boundaries: a constant feature, a single-value window, and a one-observation history.
  const constant = new FeatureStore(registry);
  constant.write("c1", "clicks_7d", 0, 7);
  assert.strictEqual(constant.asOf("c1", "clicks_7d", 0).value, 7);
  assert.strictEqual(constant.asOf("c1", "spend_30d", 0), null);
  assert.strictEqual(constant.asOf("nobody", "clicks_7d", 10), null);
  const flat = new Array(50).fill(2.5);
  assert.ok(psi(flat, flat, selectBins(flat, 4)) < 1e-12, "a constant feature cannot drift");
  const oneBin = new DriftMonitor({ threshold: 0.1, binCount: 1 });
  const degenerate = oneBin.check("flat", [1, 2, 3], [4, 5, 6]);
  assert.strictEqual(degenerate.psi, 0, "a single bin cannot separate any two windows");
  assert.strictEqual(degenerate.alert, false);
  const twoBin = new DriftMonitor({ threshold: 0.1, binCount: 2 });
  const disjoint = twoBin.check("flat", [1, 2, 3], [4, 5, 6]);
  assert.ok(disjoint.psi > 0.1 && disjoint.alert === true, "disjoint windows must alert");

  // 7. Feature redefinition bumps the version and keeps the lineage edge.
  const versions = new FeatureRegistry();
  assert.strictEqual(versions.define("f", { source: "s" }), 1);
  assert.strictEqual(versions.define("f", { source: "s", transform: "zscore" }), 2);
  assert.strictEqual(versions.features.get("f").version, 2);
  assert.deepStrictEqual(versions.impactOf("s"), ["f"]);
  assert.strictEqual(versions.topologicalOrder().length, 2);

  // 8. Invalid input is rejected, including lineage cycles.
  assert.throws(() => store.write("e0", "ghost", 0, 1), /unknown feature/);
  assert.throws(() => store.write("e0", "clicks_7d", 0, 1), /non-decreasing/);
  assert.throws(() => store.write("e0", "clicks_7d", 10, Number.NaN), /finite/);
  assert.throws(() => store.write("e0", "clicks_7d", -1, 1), /non-negative integers/);
  assert.throws(() => registry.closure("ghost", "down"), /unknown lineage node/);
  assert.throws(() => registry.addEdge("risk", "events_raw"), /cycle/);
  assert.throws(() => psi([], [1], [0.5]), /must not be empty/);
  assert.throws(() => store.batchAsOf([{ entity: "e0", asOf: 1 }], ["ghost"]), /unknown feature/);

  // 9. Determinism: the same seed rebuilds the same store contents and drift scores.
  const rerun = () => {
    const p = new Prng(7);
    const local = new FeatureStore(registry);
    for (let k = 0; k < 20; k++) local.write("d1", "clicks_7d", k, p.int(100));
    const a = Array.from({ length: 50 }, () => p.normal(1));
    const b = Array.from({ length: 50 }, () => p.normal(1) + 1);
    return { lookups: [local.asOf("d1", "clicks_7d", 7), local.asOf("d1", "clicks_7d", 19)],
      drift: new DriftMonitor({ binCount: 5 }).check("clicks_7d", a, b) };
  };
  assert.deepStrictEqual(rerun(), rerun());

  console.log(`feature store: ${store.writes} writes, ${comparisons} as-of comparisons, ` +
    `stable PSI ${calm.psi.toFixed(4)}, shifted PSI ${alarmed.psi.toFixed(4)}, ` +
    `impact of events_raw = ${registry.impactOf("events_raw").join("+")}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, FeatureRegistry, FeatureStore, psi, histogram, selectBins, DriftMonitor };
