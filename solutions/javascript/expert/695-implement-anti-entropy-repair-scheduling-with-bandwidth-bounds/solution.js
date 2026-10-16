/** Task 695 - Implement anti-entropy repair scheduling with bandwidth bounds. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Model: two parts that make up anti-entropy repair in a replicated store.
 *
 * 1. Divergence detection with a Merkle tree. Each replica builds a binary tree over its sorted
 *    key/value pairs (a leaf holds `leafSize` pairs, the hash is an FNV-1a digest of the leaf's
 *    contents and every internal node hashes its children). Two replicas compare roots and descend
 *    only into subtrees whose hash differs, so the cost is proportional to the *divergence* plus a
 *    logarithmic prefix, not to the size of the data set - the node-visit counter below makes that
 *    measurable. The set of differing keys found this way is cross-checked against a linear scan.
 *
 * 2. Repair scheduling under a bandwidth bound. Every differing key is a repair task whose weight is
 *    the number of bytes that must be sent to fetch it from the replica that holds the newer value.
 *    Rounds are bandwidth bins of capacity B, so the problem is to pack the tasks into as few rounds
 *    as possible. The heuristic is LPT list scheduling (heaviest task first into the round with the
 *    least load so far); its makespan is compared against the arithmetic lower bound
 *    max(ceil(total / B), max weight) and against an exact branch-and-bound optimum on small
 *    instances, and the bandwidth invariant (no round ever exceeds B) is checked for every round.
 *
 * Simplified versus a real system: no network latency or per-peer sessions, weights are the raw value
 * sizes, the repair direction is decided per key by a version counter, there is no concurrent third
 * replica, and the exact optimum is only computed for small task sets. Verified: hash sensitivity,
 * Merkle diff == linear scan oracle, fewer node visits than a full scan when divergence is small,
 * transfer volume below a full copy, repair convergence (equal roots after the repair, empty diff on
 * the second pass), the bandwidth invariant and load accounting, LPT within a factor of two of the
 * exact optimum and never better than it, infeasible single-task handling, and determinism.
 */
"use strict";

const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

/** FNV-1a over a string, as an unsigned 32-bit value. */
function fnv1a(text) {
  let hash = FNV_OFFSET;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  return hash >>> 0;
}

function hashLeaf(entries) {
  return fnv1a(entries.map(([key, value]) => `${key}=${value}`).join("|"));
}

function hashNode(left, right) {
  return fnv1a(`${left}#${right}`);
}

/**
 * Anti-entropy Merkle tree over a *fixed* key-space partition: the key space is split into `buckets`
 * leaf ranges (a power of two, so the tree is a full binary tree with the same shape at every
 * replica), a key falls into bucket `fnv1a(key) % buckets`, and each leaf hashes the entries in its
 * own bucket. Because the shape does not depend on which keys are present, a missing or added key
 * only perturbs the leaf that owns it, which is what makes recursive descent sound.
 */
class MerkleTree {
  constructor(entries, { buckets = 8 } = {}) {
    if (!Number.isInteger(buckets) || buckets < 2 || (buckets & (buckets - 1)) !== 0) {
      throw new RangeError("buckets must be a power of two >= 2");
    }
    this.buckets = buckets;
    this.entries = [...entries].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    const leaves = Array.from({ length: buckets }, () => []);
    for (const entry of this.entries) leaves[fnv1a(entry[0]) % buckets].push(entry);
    this.nodes = 2 * buckets - 1;
    this.root = this.build(leaves, 0, buckets);
  }

  build(leaves, start, end) {
    if (end - start === 1) {
      return { hash: hashLeaf(leaves[start]), bucket: start, entries: leaves[start], left: null, right: null };
    }
    const middle = start + Math.floor((end - start) / 2);
    const left = this.build(leaves, start, middle);
    const right = this.build(leaves, middle, end);
    return { hash: hashNode(left.hash, right.hash), bucket: -1, entries: null, left, right };
  }

  /** Keys whose value differs, or that are present at only one replica, via recursive descent. */
  diff(other, counter = { visited: 0 }) {
    const differing = new Set();
    const walk = (a, b) => {
      counter.visited++;
      if (a.hash === b.hash) return;
      if (a.left === null) {
        const leftKeys = new Map(a.entries);
        const rightKeys = new Map(b.entries);
        for (const [key, value] of leftKeys) if (rightKeys.get(key) !== value) differing.add(key);
        for (const key of rightKeys.keys()) if (!leftKeys.has(key)) differing.add(key);
        return;
      }
      walk(a.left, b.left);
      walk(a.right, b.right);
    };
    walk(this.root, other.root);
    return { differing, visited: counter.visited };
  }
}

/** Oracle: compare two key/value maps directly. */
function linearDiff(leftEntries, rightEntries) {
  const left = new Map(leftEntries);
  const right = new Map(rightEntries);
  const differing = new Set();
  for (const [key, value] of left) if (right.get(key) !== value) differing.add(key);
  for (const key of right.keys()) if (!left.has(key)) differing.add(key);
  return differing;
}

/** Repair tasks: one per differing key, weighted by the newer value's size in bytes. */
function buildTasks(differingKeys, sizes, version) {
  return [...differingKeys].sort().map((key) => ({
    key,
    weight: sizes.get(key) || 0,
    newer: version.get(key) === "left" ? "left" : "right",
  }));
}

function totalWeight(tasks) {
  return tasks.reduce((acc, task) => acc + task.weight, 0);
}

/** Lower bound on the number of rounds: the heavier of the volume bound and the largest task. */
function lowerBound(tasks, bandwidth) {
  const total = totalWeight(tasks);
  const largest = tasks.reduce((acc, task) => Math.max(acc, task.weight), 0);
  return Math.max(Math.ceil(total / bandwidth), largest > 0 ? 1 : 0, largest > bandwidth ? Infinity : 0);
}

/** LPT list scheduling: heaviest task first into the currently lightest round. */
function scheduleLpt(tasks, bandwidth) {
  if (!Number.isInteger(bandwidth) || bandwidth < 1) throw new RangeError("bandwidth must be a positive integer");
  for (const task of tasks) {
    if (task.weight < 0) throw new RangeError("task weights must not be negative");
    if (task.weight > bandwidth) throw new RangeError(`task ${task.key} is larger than the bandwidth bound`);
  }
  const rounds = [];
  for (const task of [...tasks].sort((a, b) => b.weight - a.weight || (a.key < b.key ? -1 : 1))) {
    let best = -1;
    for (let i = 0; i < rounds.length; i++) {
      if (best === -1 || rounds[i].load < rounds[best].load) best = i;
    }
    if (best === -1 || rounds[best].load + task.weight > bandwidth) {
      rounds.push({ load: task.weight, tasks: [task] });
      continue;
    }
    rounds[best].load += task.weight;
    rounds[best].tasks.push(task);
  }
  return { rounds, makespan: rounds.length };
}

/** Exact optimum by branch and bound: try every round count from the lower bound upwards. */
function optimalMakespan(tasks, bandwidth) {
  const weights = tasks.map((task) => task.weight).sort((a, b) => b - a);
  if (weights.length === 0) return 0;
  if (weights[0] > bandwidth) return Infinity;
  const start = Math.max(1, Math.ceil(totalWeight(tasks) / bandwidth));
  const feasible = (roundCount) => {
    const loads = new Array(roundCount).fill(0);
    const place = (index) => {
      if (index === weights.length) return true;
      const weight = weights[index];
      const seen = new Set();
      for (let round = 0; round < roundCount; round++) {
        if (loads[round] + weight > bandwidth) continue;
        if (seen.has(loads[round])) continue;          // symmetry: identical load levels are equivalent
        seen.add(loads[round]);
        loads[round] += weight;
        if (place(index + 1)) return true;
        loads[round] -= weight;
      }
      return false;
    };
    return place(0);
  };
  for (let count = start; count <= weights.length; count++) {
    if (feasible(count)) return count;
  }
  return Infinity;
}

function maxRoundLoad(schedule) {
  return schedule.rounds.reduce((acc, round) => Math.max(acc, round.load), 0);
}

/** Apply the repairs: the newer value of every differing key wins. */
function applyRepair(leftEntries, rightEntries, tasks) {
  const left = new Map(leftEntries);
  const right = new Map(rightEntries);
  for (const task of tasks) {
    if (task.newer === "left") right.set(task.key, left.get(task.key));
    else left.set(task.key, right.get(task.key));
  }
  return [[...left.entries()], [...right.entries()]];
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  int(bound) {
    return this.next() % bound;
  }
}

module.exports = {
  fnv1a, MerkleTree, linearDiff, buildTasks, totalWeight, lowerBound, scheduleLpt,
  optimalMakespan, maxRoundLoad, applyRepair, Lcg,
};

if (require.main === module) {
  const assert = require("node:assert");

  const lcg = new Lcg(695);

  function replica(count, keyWidth, valueSize, salt) {
    const entries = [];
    const sizes = new Map();
    for (let i = 0; i < count; i++) {
      const key = `k${String(i).padStart(keyWidth, "0")}`;
      const value = `v${salt}-${i}`.padEnd(valueSize, "x").slice(0, valueSize);
      entries.push([key, value]);
      sizes.set(key, value.length);
    }
    return { entries, sizes };
  }

  // --- hashing and tree shape ---------------------------------------------------------------
  assert.strictEqual(fnv1a(""), FNV_OFFSET);
  assert.strictEqual(fnv1a("a"), fnv1a("a"));
  assert.notStrictEqual(fnv1a("a"), fnv1a("b"));
  const a = replica(16, 3, 8, "a");
  const b = replica(16, 3, 8, "a");
  const treeA = new MerkleTree(a.entries, { buckets: 8 });
  const treeB = new MerkleTree(b.entries, { buckets: 8 });
  assert.strictEqual(treeA.root.hash, treeB.root.hash, "identical replicas share the root hash");
  assert.strictEqual(treeA.nodes, 15, "8 buckets make a full binary tree of 15 nodes, whatever the data");
  const emptyDiff = treeA.diff(treeB);
  assert.strictEqual(emptyDiff.differing.size, 0);
  assert.strictEqual(emptyDiff.visited, 1, "an identical root needs a single comparison");
  const emptyTree = new MerkleTree([], { buckets: 8 });
  assert.strictEqual(emptyTree.nodes, 15);
  assert.deepStrictEqual([...emptyTree.diff(treeA).differing].sort(), a.entries.map(([key]) => key).sort(),
    "an empty replica is behind on every key");
  assert.throws(() => new MerkleTree(a.entries, { buckets: 3 }), RangeError, "buckets must be a power of two");

  // --- the diff equals the linear-scan oracle ------------------------------------------------
  let trials = 0;
  let visitedTotal = 0;
  for (let trial = 0; trial < 12; trial++) {
    const count = 8 + lcg.int(25);
    const left = replica(count, 3, 12, "L");
    const right = replica(count, 3, 12, "L");
    const changed = 1 + lcg.int(3);
    for (let i = 0; i < changed; i++) {
      const index = lcg.int(count);
      right.entries[index] = [right.entries[index][0], `patched-${trial}-${i}`.padEnd(12, "z").slice(0, 12)];
    }
    const leftTree = new MerkleTree(left.entries, { buckets: 8 });
    const rightTree = new MerkleTree(right.entries, { buckets: 8 });
    const { differing, visited } = leftTree.diff(rightTree);
    assert.deepStrictEqual([...differing].sort(), [...linearDiff(left.entries, right.entries)].sort(),
      `oracle agreement on trial ${trial}`);
    assert.strictEqual(differing.size, changed, "exactly the patched keys differ");
    assert.ok(visited <= 2 * count, `the descent stays small (${visited} nodes for ${count} keys)`);
    visitedTotal += visited;
    trials++;
  }
  assert.strictEqual(trials, 12);
  assert.ok(visitedTotal < 12 * 2 * 20, "the Merkle descent never scans every key");

  // a replica that lost a key is detected as well
  const dropped = a.entries.filter(([key]) => key !== "k005");
  const droppedTree = new MerkleTree(dropped, { buckets: 8 });
  assert.deepStrictEqual([...treeA.diff(droppedTree).differing], ["k005"], "a missing key is a difference");

  // --- tasks and the bandwidth bound ----------------------------------------------------------
  const sizes = new Map([["k1", 5], ["k2", 4], ["k3", 4], ["k4", 3], ["k5", 3], ["k6", 2]]);
  const version = new Map([["k1", "left"], ["k2", "right"], ["k3", "left"], ["k4", "right"], ["k5", "left"], ["k6", "right"]]);
  const tasks = buildTasks(sizes.keys(), sizes, version);
  assert.strictEqual(tasks.length, 6);
  assert.deepStrictEqual(tasks.map((task) => task.weight), [5, 4, 4, 3, 3, 2], "tasks are built in key order");
  assert.strictEqual(totalWeight(tasks), 21);
  assert.strictEqual(lowerBound(tasks, 8), 3, "ceil(21 / 8) = 3 rounds at best");

  const schedule = scheduleLpt(tasks, 8);
  assert.strictEqual(maxRoundLoad(schedule), 8, "LPT fills a round exactly");
  for (const round of schedule.rounds) {
    assert.ok(round.load <= 8, "the bandwidth bound is never exceeded");
    assert.strictEqual(round.load, round.tasks.reduce((acc, task) => acc + task.weight, 0), "load accounting");
  }
  assert.strictEqual(schedule.rounds.reduce((acc, round) => acc + round.load, 0), 21, "every byte is scheduled");
  const scheduledKeys = schedule.rounds.flatMap((round) => round.tasks.map((task) => task.key)).sort();
  assert.deepStrictEqual(scheduledKeys, [...sizes.keys()].sort(), "each task is scheduled exactly once");
  assert.strictEqual(schedule.makespan, 3, "and 3 is the optimum here");
  assert.strictEqual(optimalMakespan(tasks, 8), 3, "the exhaustive search agrees");
  assert.strictEqual(scheduleLpt([], 8).makespan, 0, "nothing to repair needs no rounds");
  assert.strictEqual(lowerBound([], 8), 0);
  assert.throws(() => scheduleLpt([{ key: "big", weight: 9 }], 8), RangeError, "a task larger than B");
  assert.strictEqual(optimalMakespan([{ key: "big", weight: 9 }], 8), Infinity);
  assert.strictEqual(scheduleLpt([{ key: "one", weight: 8 }], 8).makespan, 1, "an exactly full round");

  // --- LPT against the exact optimum on random instances -------------------------------------
  let optimalCases = 0;
  let tightCases = 0;
  for (let trial = 0; trial < 60; trial++) {
    const taskCount = 3 + lcg.int(6);
    const bandwidth = 10 + lcg.int(15);
    const random = [];
    for (let i = 0; i < taskCount; i++) random.push({ key: `t${i}`, weight: 1 + lcg.int(bandwidth) });
    const heuristic = scheduleLpt(random, bandwidth);
    const optimum = optimalMakespan(random, bandwidth);
    assert.ok(optimum !== Infinity);
    assert.ok(heuristic.makespan >= optimum, "a heuristic can never beat the optimum");
    assert.ok(heuristic.makespan <= 2 * optimum, `LPT within a factor 2 (${heuristic.makespan} vs ${optimum})`);
    assert.ok(maxRoundLoad(heuristic) <= bandwidth, "bandwidth invariant");
    assert.ok(heuristic.makespan >= lowerBound(random, bandwidth), "and the lower bound is respected");
    if (heuristic.makespan === optimum) tightCases++;
    optimalCases++;
  }
  assert.strictEqual(optimalCases, 60);
  assert.ok(tightCases >= 50, `LPT is optimal on most instances (${tightCases}/60)`);

  // --- repair convergence --------------------------------------------------------------------
  const left = replica(24, 3, 10, "same");
  const right = replica(24, 3, 10, "same");
  right.entries[3] = ["k003", "newer-003".padEnd(10, "n")];
  right.entries[17] = ["k017", "newer-017".padEnd(10, "n")];
  deleteSizes(right);
  function deleteSizes(instance) {
    instance.sizes = new Map(instance.entries.map(([key, value]) => [key, value.length]));
  }
  const leftTree = new MerkleTree(left.entries, { buckets: 8 });
  const rightTree = new MerkleTree(right.entries, { buckets: 8 });
  const { differing } = leftTree.diff(rightTree);
  assert.deepStrictEqual([...differing].sort(), ["k003", "k017"]);
  const repairTasks = buildTasks(differing, right.sizes,
    new Map([...differing].map((key) => [key, "right"])));
  assert.strictEqual(totalWeight(repairTasks), 20, "two ten-byte values must be fetched");
  const fullCopyBytes = right.entries.reduce((acc, [, value]) => acc + value.length, 0);
  assert.ok(totalWeight(repairTasks) < fullCopyBytes / 10, "anti-entropy transfers a fraction of a full copy");
  const [repairedLeft, repairedRight] = applyRepair(left.entries, right.entries, repairTasks);
  const repairedTreeLeft = new MerkleTree(repairedLeft, { buckets: 8 });
  const repairedTreeRight = new MerkleTree(repairedRight, { buckets: 8 });
  assert.strictEqual(repairedTreeLeft.root.hash, repairedTreeRight.root.hash, "the roots converge");
  assert.strictEqual(repairedTreeLeft.diff(repairedTreeRight).differing.size, 0, "and the second pass is empty");
  const repairSchedule = scheduleLpt(repairTasks, 16);
  assert.strictEqual(repairSchedule.makespan, 2, "20 bytes at 16 per round needs two rounds");
  assert.strictEqual(optimalMakespan(repairTasks, 16), 2);
  const singleRound = scheduleLpt(repairTasks, 32);
  assert.strictEqual(singleRound.makespan, 1, "a wider pipe needs one round");
  assert.strictEqual(maxRoundLoad(singleRound), 20);
  assert.strictEqual(lowerBound(repairTasks, 16), 2, "the volume bound");
  assert.strictEqual(lowerBound(repairTasks, 100), 1, "and the bound drops with bandwidth");

  // --- merging a third replica into the schedule ---------------------------------------------
  const third = replica(24, 3, 10, "same");
  third.entries[5] = ["k005", "third-005".padEnd(10, "t")];
  const thirdTree = new MerkleTree(third.entries, { buckets: 8 });
  const merged = [];
  for (const other of [rightTree, thirdTree]) {
    const { differing: keys } = repairedTreeRight.diff(other);
    merged.push(...buildTasks(keys, new Map(other.entries.map(([key, value]) => [key, value.length])),
      new Map([...keys].map((key) => [key, "right"]))));
  }
  assert.deepStrictEqual(merged.map((task) => task.key), ["k003", "k005", "k017"],
    "the third replica is behind on the two repaired keys and ahead on its own");
  assert.strictEqual(totalWeight(merged), 30);
  const mergedSchedule = scheduleLpt(merged, 32);
  assert.strictEqual(mergedSchedule.makespan, 1, "30 bytes fit in one round at 32 per round");
  assert.strictEqual(scheduleLpt(merged, 16).makespan, 3, "and need three rounds at half the bandwidth");

  // --- determinism ---------------------------------------------------------------------------
  const replayA = scheduleLpt(tasks, 8);
  const replayB = scheduleLpt(buildTasks(sizes.keys(), sizes, version), 8);
  assert.deepStrictEqual(replayB.rounds.map((round) => round.tasks.map((task) => task.key)),
    replayA.rounds.map((round) => round.tasks.map((task) => task.key)));
  assert.strictEqual(new MerkleTree(a.entries, { buckets: 8 }).root.hash, treeA.root.hash);

  console.log(`trials=${trials} optimal=${tightCases}/60 rounds=${schedule.makespan}`);
  console.log("PASS");
}
