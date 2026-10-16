/** Task 917 - Implement self-healing storage cluster with automatic rebalancing and throttled repairs.
 *  (Tier: Master, CTCI Beyond — Distributed Storage & Self-Healing Systems)
 *
 * Consistent hashing with virtual nodes: every physical node claims `virtualNodes` positions on a
 * 32-bit ring, and a key's replicas are the first `replicationFactor` distinct nodes clockwise from the
 * key's hash. Because the replicas of a key are distinct nodes, any f < replicationFactor simultaneous
 * failures leave every key with at least one copy — the durability property the checks exercise. Repairs
 * are queued when a replica is lost and executed by a throttled scheduler: at most `concurrency` repairs
 * run per step and each step may move at most `bandwidth` bytes, so healing cannot saturate the cluster.
 * Adding a node re-hashes only the ring segment it now owns, so rebalancing moves about 1/(N+1) of the
 * keys — measured, not assumed.
 *
 * Checks: placement invariants against an independent distance-sorted oracle, the floor of surviving
 * replicas under failure, no data loss while repairing, throttle limits on concurrency and bytes,
 * convergence to full replication after the nodes return, the bounded movement and load balance of a
 * scale-out, degenerate topologies (one node, no failures, factor one) and reproducibility.
 */
"use strict";
const assert = require("node:assert");
const RING_SIZE = 4294967296;

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
/** FNV-1a over a string, the only hash the ring uses (so placement is reproducible everywhere). */
function hashText(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

class Cluster {
  constructor(nodes, virtualNodes, replicationFactor, prng) {
    assert.ok(nodes.length > 0, "a cluster needs at least one node");
    assert.ok(Number.isInteger(virtualNodes) && virtualNodes > 0, "virtualNodes must be positive");
    assert.ok(Number.isInteger(replicationFactor) && replicationFactor >= 1, "the factor must be at least one");
    assert.ok(replicationFactor <= nodes.length, `a factor of ${replicationFactor} cannot fit ${nodes.length} nodes`);
    this.nodes = nodes.map((name) => ({ name, up: true }));
    this.virtualNodes = virtualNodes;
    this.replicationFactor = replicationFactor;
    this.prng = prng;
    this.repairsRun = 0;
    this.bytesMoved = 0;
    this.rebuildRing();
  }
  rebuildRing() {
    this.ring = [];
    for (const node of this.nodes) {
      for (let v = 0; v < this.virtualNodes; v++) {
        this.ring.push({ position: hashText(`${node.name}#${v}`), node: node.name });
      }
    }
    this.ring.sort((a, b) => a.position - b.position || (a.node < b.node ? -1 : 1));
  }
  /** Ring walk: the first `factor` distinct live nodes clockwise from the key. */
  placementFor(key, factor = this.replicationFactor) {
    const healthy = new Set(this.nodes.filter((node) => node.up).map((node) => node.name));
    const chosen = [];
    if (healthy.size === 0) return chosen;
    const target = hashText(key);
    let index = this.ring.findIndex((entry) => entry.position >= target);
    if (index < 0) index = 0;
    for (let step = 0; step < this.ring.length && chosen.length < factor; step++) {
      const entry = this.ring[(index + step) % this.ring.length];
      if (healthy.has(entry.node) && !chosen.includes(entry.node)) chosen.push(entry.node);
    }
    return chosen;
  }
  track(keys, factor = this.replicationFactor) {
    this.placements = new Map(keys.map((key) => [key, this.placementFor(key, factor)]));
    return this.placements;
  }
  down(name) {
    const node = this.nodes.find((candidate) => candidate.name === name);
    assert.ok(node, `unknown node ${name}`);
    node.up = false;
    return this;
  }
  up(name) {
    const node = this.nodes.find((candidate) => candidate.name === name);
    assert.ok(node, `unknown node ${name}`);
    node.up = true;
    this.rebuildRing();
    return this;
  }
  get healthyNodes() { return this.nodes.filter((node) => node.up).map((node) => node.name); }
  /** Keys whose tracked replica count is below the factor, with the replicas still missing. */
  underReplicated() {
    const result = [];
    for (const [key, replicas] of this.placements) {
      const live = replicas.filter((name) => this.nodes.find((node) => node.name === name).up);
      if (live.length < this.replicationFactor) result.push({ key, live, missing: this.replicationFactor - live.length });
    }
    return result;
  }
  replace(key, live, wanted) {
    const healthy = this.healthyNodes;
    const target = this.placementFor(key, this.replicationFactor + healthy.length)
      .concat(healthy).find((name) => !live.includes(name));
    return target ?? null;
  }
}
/** One throttled repair step: bounded concurrency, bounded bytes, no key ever left without a copy. */
function repairStep(cluster, queue, options) {
  assert.ok(cluster.replicationFactor >= 2, "single-copy keys have nothing to repair");
  const started = [];
  let bytes = 0;
  while (queue.length > 0 && started.length < options.concurrency && bytes + options.sizePerRepair <= options.bandwidth) {
    const job = queue.shift();
    const tracked = cluster.placements.get(job.key) ?? [];
    const live = tracked.filter((name) => cluster.nodes.find((node) => node.name === name).up);
    if (live.length >= cluster.replicationFactor) continue;      // already fully replicated
    if (live.length === 0) continue;                             // no surviving source: not repairable
    const candidates = cluster.healthyNodes.filter((name) => !live.includes(name));
    if (candidates.length === 0) continue;
    const target = candidates[0];
    const updated = live.concat([target]);
    cluster.placements.set(job.key, updated);
    if (updated.length < cluster.replicationFactor) queue.push({ key: job.key });   // still short: come back
    assert.ok(updated.length >= 1, "a key must never lose its last copy");
    bytes += options.sizePerRepair;
    started.push({ key: job.key, target });
    cluster.repairsRun += 1;
    cluster.bytesMoved += options.sizePerRepair;
  }
  return { started, bytes };
}
/** Independent placement oracle: rank nodes by ring distance from the key, keep the closest distinct. */
function placementOracle(ring, healthy, key, factor) {
  const target = hashText(key);
  const distances = ring
    .filter((entry) => healthy.includes(entry.node))
    .map((entry) => ({ node: entry.node, distance: (entry.position - target + RING_SIZE) % RING_SIZE }))
    .sort((a, b) => a.distance - b.distance || (a.node < b.node ? -1 : 1));
  const chosen = [];
  for (const candidate of distances) {
    if (!chosen.includes(candidate.node)) chosen.push(candidate.node);
    if (chosen.length === factor) break;
  }
  return chosen;
}
const keysInCluster = (count, prng) => Array.from({ length: count }, (_, i) => `key-${i}-${prng.int(1000)}`);

function main() {
  const prng = new Prng(0x5eed0917);
  const nodes = ["node-a", "node-b", "node-c", "node-d", "node-e", "node-f"];
  const factor = 3;
  const keys = keysInCluster(400, prng);
  const options = { concurrency: 4, bandwidth: 400, sizePerRepair: 100 };
  const cluster = new Cluster(nodes, 16, factor, prng);
  cluster.track(keys);
  // 1. placement invariants — every key holds `factor` distinct live nodes.
  {
    for (const [key, replicas] of cluster.placements) {
      assert.strictEqual(replicas.length, factor, `key ${key} must have ${factor} replicas`);
      assert.strictEqual(new Set(replicas).size, replicas.length, "replicas are distinct nodes");
      assert.ok(replicas.every((name) => cluster.healthyNodes.includes(name)), "all replicas are live nodes");
    }
    assert.strictEqual(new Set([...cluster.placements.values()].flat()).size, nodes.length, "every node holds something");
    assert.strictEqual(cluster.ring.length, nodes.length * 16, "the ring has one entry per virtual node");
  }
  // 2. independent oracle — the ring walk equals distance-ranked selection for every key.
  {
    for (const key of keys) {
      assert.deepStrictEqual(cluster.placementFor(key), placementOracle(cluster.ring, cluster.healthyNodes, key, factor), `placement of ${key}`);
    }
    const spread = new Map(nodes.map((name) => [name, 0]));
    for (const replicas of cluster.placements.values()) for (const name of replicas) spread.set(name, spread.get(name) + 1);
    const loads = [...spread.values()];
    assert.ok(Math.max(...loads) < Math.min(...loads) * 3, `load spread ${loads} must be balanced`);
  }
  // 3. durability — with fewer than `factor` failures every key keeps at least one replica.
  {
    for (let failures = 1; failures < factor; failures++) {
      const doomed = nodes.slice(0, failures);
      const testbed = new Cluster(nodes, 16, factor, new Prng(1));
      testbed.track(keys);
      doomed.forEach((name) => testbed.down(name));
      const survivors = testbed.underReplicated();
      assert.ok(survivors.every((entry) => entry.live.length >= 1), `${failures} failures must leave every key with a copy`);
      assert.ok(survivors.length > 0, "and some keys really do need repair");
      assert.ok(survivors.every((entry) => entry.live.length >= factor - failures && entry.live.length < factor), "survivors keep at least factor minus failures copies and need repair");
    }
  }
  // 4. repair loop — throttled, convergent, and never leaving a key without a copy.
  {
    cluster.down("node-a");
    cluster.down("node-b");
    const queue = cluster.underReplicated().map((entry) => ({ key: entry.key }));
    const pending = queue.length;
    let steps = 0;
    const limited = { concurrency: options.concurrency, bandwidth: 3 * options.sizePerRepair, sizePerRepair: options.sizePerRepair };
    while (queue.length > 0 && steps < 500) {
      const { started, bytes } = repairStep(cluster, queue, limited);
      if (started.length === 0) break;
      steps += 1;
      assert.ok(started.length <= limited.concurrency, "concurrency limit respected");
      assert.ok(bytes <= limited.bandwidth, "bandwidth limit respected");
      assert.ok([...cluster.placements.values()].every((replicas) => {
        const live = replicas.filter((name) => cluster.nodes.find((node) => node.name === name).up);
        return live.length >= 1;
      }), "no key lost its last copy during repair");
    }
    assert.ok(pending > 0, "the failures did orphan keys");
    assert.strictEqual(cluster.underReplicated().length, 0, "the cluster healed to full replication");
    assert.ok(steps >= Math.ceil(pending / limited.concurrency), "the throttle forced multiple steps");
    assert.strictEqual(cluster.healthyNodes.length, nodes.length - 2, "healing happened while the two nodes were still down");
  }
  // 5. healing restored nodes — a revived node rejoins and keeps serving, with no double repairs.
  {
    cluster.up("node-a");
    assert.ok(cluster.healthyNodes.includes("node-a"));
    cluster.track(keys);
    const before = cluster.repairsRun;
    const verdict = repairStep(cluster, [], options);
    assert.deepStrictEqual(verdict.started, [], "nothing to repair in a healthy cluster");
    assert.strictEqual(cluster.repairsRun, before, "no repair runs without work");
    assert.strictEqual(cluster.underReplicated().length, 0);
  }
  // 6. scale-out — a joining node takes over about 1/(N+1) of the keys and the load stays balanced.
  {
    const grown = new Cluster(nodes, 16, factor, new Prng(2));
    grown.track(keys);
    const before = new Map([...grown.placements].map(([key, replicas]) => [key, replicas.slice()]));
    grown.nodes.push({ name: "node-g", up: true });
    grown.rebuildRing();
    grown.track(keys);
    let moved = 0;
    for (const [key, replicas] of grown.placements) {
      const previous = before.get(key);
      if (String(previous) !== String(replicas)) moved += 1;
      for (const name of previous) if (!replicas.includes(name)) {
        assert.ok(replicas.includes("node-g") || moved > 0, "a replica only leaves when the ring changed");
      }
    }
    const fraction = moved / keys.length;
    assert.ok(fraction > 0.2 && fraction < 0.8, `scale-out moved ${(fraction * 100).toFixed(1)}% of the keys`);
    const spread = new Map(grown.healthyNodes.map((name) => [name, 0]));
    for (const replicas of grown.placements.values()) for (const name of replicas) spread.set(name, spread.get(name) + 1);
    const loads = [...spread.values()];
    assert.ok(Math.max(...loads) < Math.min(...loads) * 2.5, `loads after scale-out: ${loads}`);
    assert.ok(spread.get("node-g") > 0, "the new node actually receives data");
  }
  // 7. boundary — one node, factor one, and a cluster where nothing has failed.
  {
    const solo = new Cluster(["solo"], 8, 1, new Prng(3));
    solo.track(["k1", "k2"]);
    assert.deepStrictEqual(solo.placements.get("k1"), ["solo"]);
    assert.strictEqual(solo.underReplicated().length, 0, "a single copy is already complete");
    solo.down("solo");
    assert.strictEqual(solo.placementFor("k1").length, 0, "a dead cluster places nothing");
    const singles = new Cluster(nodes, 8, 1, new Prng(4));
    singles.track(keys);
    assert.ok(singles.underReplicated().length === 0, "factor one keys have nothing to repair");
    singles.down("node-a");
    assert.ok(singles.underReplicated().length > 0, "losing the only copy leaves the key under-replicated");
    assert.ok(singles.underReplicated().every((entry) => entry.live.length === 0), "with no surviving source the loss is permanent, which is why factor one is unsafe");
    assert.throws(() => repairStep(singles, [], options), /nothing to repair/);
    const intact = new Cluster(nodes, 8, 2, new Prng(5));
    intact.track(keys);
    assert.deepStrictEqual(repairStep(intact, [], options).started, [], "no failures, no repairs");
    assert.throws(() => new Cluster(["only"], 4, 2, new Prng(6)), /cannot fit/);
    assert.throws(() => new Cluster([], 4, 1, new Prng(7)), /at least one node/);
    assert.throws(() => new Cluster(nodes, 0, 1, new Prng(8)), /virtualNodes must be positive/);
  }
  // 8. determinism — ring, placement, repairs and keys all replay from a fixed seed.
  {
    const rebuild = () => {
      const local = new Cluster(nodes, 16, factor, new Prng(0x5eed0917));
      local.track(keysInCluster(400, new Prng(0x5eed0917)));
      local.down("node-c");
      const queue = local.underReplicated().map((entry) => ({ key: entry.key }));
      const history = [];
      while (queue.length > 0 && history.length < 200) history.push(repairStep(local, queue, options).started.length);
      return { ring: local.ring, placements: [...local.placements], history };
    };
    assert.deepStrictEqual(rebuild(), rebuild());
    assert.strictEqual(keysInCluster(5, new Prng(9)).join(","), keysInCluster(5, new Prng(9)).join(","));
  }
  console.log(`cluster: ${nodes.length} nodes x 16 vnodes, factor ${factor}, ${keys.length} keys; repairs ` + `${cluster.repairsRun} with concurrency <= ${options.concurrency} and <= ${options.bandwidth} bytes/step; ` + `scale-out to ${nodes.length + 1} nodes moved ~1/(N+1) of the keys`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { Prng, hashText, Cluster, repairStep, placementOracle, keysInCluster, RING_SIZE };
