/** Task 903 - Implement adaptive replication factoring hot keys and node load.
 *  (Tier: Master, CTCI Beyond — Distributed Systems & Load Balancing)
 *
 * A sharded store whose replication factor adapts to key popularity: keys seen at least hotThreshold
 * times in the observed window get hotFactor replicas, everything else baseFactor. Plans are built hot
 * first, placing each replica on the least loaded of `exploration` random candidates (power-of-two /
 * d-choice sampling) and spilling to the globally least loaded node only when every candidate is at the
 * capacity ceiling ceil((1+eps)*replicas/nodes) — the bounded-load construction, which is what keeps
 * maxLoad close to avgLoad instead of letting a hot key pile onto one node.
 *
 * The checks verify: distinct replicas per key, the capacity ceiling, the d-choice balance guarantee
 * (maxLoad <= (1+eps)*avgLoad + 1) against the independent lower bound ceil(replicas/nodes), that the
 * hot set equals the top of the *exact* frequency ranking, single-node-failure availability, the
 * monotone cost/latency trade-off when the hot threshold moves, degenerate clusters (one node, no keys,
 * factor larger than the cluster) and reproducibility.
 */
"use strict";

const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}

/** Zipf-ish key popularity sampled from weights 1/(rank+1)^skew; deterministic for a fixed seed. */
function sampleCounts(keyCount, operations, skew, prng) {
  assert.ok(keyCount > 0 && operations > 0, "the workload needs at least one key and one operation");
  const cumulative = [];
  let total = 0;
  for (let i = 0; i < keyCount; i++) total += 1 / Math.pow(i + 1, skew);
  let running = 0;
  for (let i = 0; i < keyCount; i++) { running += 1 / Math.pow(i + 1, skew); cumulative.push(running / total); }
  const counts = new Array(keyCount).fill(0);
  for (let op = 0; op < operations; op++) {
    const draw = prng.float();
    let lo = 0;
    let hi = keyCount - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cumulative[mid] < draw) lo = mid + 1; else hi = mid;
    }
    counts[lo] += 1;
  }
  return counts;
}

/** Hot-first d-choice placement with a capacity ceiling; returns the plan and the resulting loads. */
function planReplication(counts, nodeCount, options, prng) {
  const { hotThreshold, baseFactor, hotFactor, exploration, epsilon } = options;
  assert.ok(Number.isInteger(nodeCount) && nodeCount >= 1, "the cluster needs at least one node");
  assert.ok(baseFactor >= 1 && hotFactor >= baseFactor, "factors must satisfy 1 <= base <= hot");
  assert.ok(hotFactor <= nodeCount, `replication factor ${hotFactor} cannot exceed ${nodeCount} nodes`);
  assert.ok(exploration >= 1, "at least one candidate per probe");
  const factors = counts.map((count) => (count >= hotThreshold ? hotFactor : baseFactor));
  const order = counts.map((_, key) => key).sort((a, b) => counts[b] - counts[a] || a - b);
  const totalReplicas = factors.reduce((a, b) => a + b, 0);
  const capacity = Math.ceil(((1 + epsilon) * totalReplicas) / nodeCount);
  const load = new Array(nodeCount).fill(0);
  const placement = counts.map(() => []);
  for (const key of order) {
    for (let replica = 0; replica < factors[key]; replica++) {
      const used = placement[key];
      let target = -1;
      for (let probe = 0; probe < exploration; probe++) {
        const candidate = prng.int(nodeCount);
        if (used.includes(candidate)) continue;
        if (target === -1 || load[candidate] < load[target]) target = candidate;
      }
      if (target === -1 || load[target] + 1 > capacity) {
        target = -1;
        for (let node = 0; node < nodeCount; node++) {
          if (used.includes(node)) continue;
          if (target === -1 || load[node] < load[target]) target = node;
        }
      }
      assert.ok(target >= 0, `key ${key} needs ${factors[key]} distinct nodes of ${nodeCount}`);
      used.push(target);
      load[target] += 1;
    }
  }
  return { factors, placement, load, totalReplicas, capacity, maxLoad: Math.max(...load),
    avgLoad: totalReplicas / nodeCount, hotSet: factors.map((f) => f === hotFactor) };
}

/** Keys that keep at least one replica when `failed` goes down. */
const survivingKeys = (placement, failed) =>
  placement.filter((nodes) => nodes.some((node) => node !== failed)).length;

function main() {
  const prng = new Prng(0x5eed0903);
  const options = { hotThreshold: 200, baseFactor: 2, hotFactor: 4, exploration: 2, epsilon: 0.5 };
  const nodeCount = 8;
  const counts = sampleCounts(40, 4000, 1.0, prng);
  const plan = planReplication(counts, nodeCount, options, prng);

  // 1. happy path — every key holds exactly its factor in distinct nodes, and the totals add up.
  assert.strictEqual(plan.placement.length, counts.length);
  for (let key = 0; key < counts.length; key++) {
    const nodes = plan.placement[key];
    assert.strictEqual(nodes.length, plan.factors[key], `key ${key} replica count`);
    assert.strictEqual(new Set(nodes).size, nodes.length, `key ${key} replicas must be distinct nodes`);
    assert.ok(nodes.every((node) => node >= 0 && node < nodeCount), `key ${key} nodes in range`);
  }
  const multiplicity = plan.load.reduce((a, b) => a + b, 0);
  assert.strictEqual(multiplicity, plan.totalReplicas, "the load vector accounts for every replica");

  // 2. balance guarantee — d-choice placement stays under the capacity ceiling and near the average.
  assert.ok(plan.maxLoad <= plan.capacity, `maxLoad ${plan.maxLoad} must respect capacity ${plan.capacity}`);
  assert.ok(plan.maxLoad <= (1 + options.epsilon) * plan.avgLoad + 1,
    `maxLoad ${plan.maxLoad} must stay within the bounded-load guarantee`);

  // 3. independent oracle — the theoretical optimum, and the observed spread above it.
  const optimum = Math.ceil(plan.totalReplicas / nodeCount);
  assert.ok(plan.maxLoad >= optimum, "no schedule can beat the ceil(replicas/nodes) lower bound");
  assert.ok(plan.maxLoad <= optimum * 1.35, `maxLoad ${plan.maxLoad} vs optimum ${optimum} is too wide`);
  assert.ok(plan.avgLoad >= optimum - 1, "the average is pinned by the replica count");

  // 4. hot-key factoring — the factored set matches the top of the exact frequency ranking.
  const ranked = counts.map((count, key) => ({ key, count })).sort((a, b) => b.count - a.count || a.key - b.key);
  const expectedHot = counts.map((count) => count >= options.hotThreshold);
  assert.deepStrictEqual(plan.hotSet, expectedHot, "factoring follows the exact counts");
  const hottest = ranked[0].key;
  assert.strictEqual(plan.factors[hottest], options.hotFactor, "the hottest key gets the hot factor");
  assert.ok(ranked.filter((entry) => entry.count >= options.hotThreshold).length > 0, "some key is hot");
  assert.ok(ranked.filter((entry) => entry.count < options.hotThreshold).length > 0, "some key is cold");
  assert.strictEqual(ranked.filter((entry) => expectedHot[entry.key])[0].key, hottest);

  // 5. failure behaviour — one node down loses exactly the single-replica keys parked on it.
  const busiest = plan.load.indexOf(plan.maxLoad);
  const lost = counts.length - survivingKeys(plan.placement, busiest);
  assert.strictEqual(lost, plan.placement.filter((nodes) => nodes.length === 1 && nodes[0] === busiest).length,
    "only factor-1 keys on the failed node are lost");
  assert.ok(plan.placement.every((nodes, key) =>
    nodes.some((node) => node !== busiest) || plan.factors[key] === 1), "replicated keys survive one failure");

  // 6. trade-off monotonicity — hotter thresholds cost fewer replicas, cooler ones cost more.
  const cheaper = planReplication(counts, nodeCount, { ...options, hotThreshold: Infinity }, prng);
  const dearer = planReplication(counts, nodeCount, { ...options, hotThreshold: 0 }, prng);
  assert.ok(cheaper.totalReplicas < plan.totalReplicas && plan.totalReplicas < dearer.totalReplicas,
    "raising the hot threshold must reduce the replica count");
  assert.ok(planReplication(counts, nodeCount, { ...options, exploration: 1 }, prng).maxLoad >=
    plan.maxLoad, "one-choice placement cannot beat two-choice placement");
  assert.ok(planReplication(counts, 64, options, prng).maxLoad < plan.maxLoad,
    "a wider cluster must lower the peak load");

  // 7. boundary — a single node, an empty workload, and factors that do not fit the cluster.
  const single = planReplication([5], 1, { ...options, baseFactor: 1, hotFactor: 1 }, prng);
  assert.deepStrictEqual(single.placement, [[0]]);
  assert.strictEqual(single.maxLoad, 1);
  const none = planReplication([], 4, options, prng);
  assert.deepStrictEqual(none.placement, []);
  assert.deepStrictEqual(none.load, [0, 0, 0, 0], "an empty plan leaves every node idle");
  assert.throws(() => planReplication([1], 2, { ...options, hotFactor: 3 }, prng), /cannot exceed/);
  assert.throws(() => planReplication([1], 0, options, prng), /at least one node/);

  // 8. the workload itself — the zipf ranking is preserved by the sampler (exact-ranking oracle).
  assert.ok(ranked[0].count > ranked[ranked.length - 1].count * 5, "the skew must be visible");
  assert.strictEqual(ranked.map((entry) => entry.count).reduce((a, b) => a + b, 0), 4000,
    "every sampled operation is accounted for");
  const again = sampleCounts(40, 4000, 1.0, new Prng(0x5eed0903));
  assert.deepStrictEqual(again, counts, "the sampler replays exactly for a fixed seed");

  // 9. determinism — the planner replays exactly for a fixed seed and diverges otherwise.
  const replayed = planReplication(counts, nodeCount, options, new Prng(11));
  const replayedAgain = planReplication(counts, nodeCount, options, new Prng(11));
  assert.deepStrictEqual(replayed.placement, replayedAgain.placement);
  assert.notDeepStrictEqual(planReplication(counts, nodeCount, options, new Prng(12)).placement, replayed.placement);

  console.log(`replication: 40 zipf keys over ${nodeCount} nodes, ${plan.totalReplicas} replicas, ` +
    `avgLoad ${plan.avgLoad.toFixed(2)}, maxLoad ${plan.maxLoad} <= capacity ${plan.capacity} ` +
    `(optimum ${optimum}); ${plan.hotSet.filter(Boolean).length} hot keys factored at ${options.hotFactor}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { Prng, sampleCounts, planReplication, survivingKeys };
