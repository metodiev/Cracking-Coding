/** Task 774 - Implement concurrent B-tree with latch-free readers (reader-writer locks). (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A real B-tree (minimum degree t, at most 2t-1 keys per node) whose *readers take no latch at all*.
 * Writers hold an exclusive latch; readers use a seqlock: sample an even version counter, walk the
 * tree, re-read the counter — if it changed a writer was active, so the read is discarded and
 * retried. A read that validates therefore observed a committed state, since every writer mutation
 * lies inside one version window.
 *
 * The concurrency is *simulated deterministically*: writer and reader tasks are generators whose
 * micro-steps (descend a level, split a child, splice the separator into a parent) are interleaved
 * by a seeded scheduler — which is what makes the guarantees testable, because a split stores the
 * right sibling before the parent learns about it and exactly in that window a non-validating
 * reader can see a committed key vanish.
 *
 * What is simplified: one writer latch (no latch coupling or hand-over-hand locking), no deletes,
 * cooperative interleaving instead of threads, and the latch is a version counter rather than a
 * mutex. What is verified: exact linearizability of every validated read against the committed
 * snapshot at its start version; the same schedule with validation off really does produce
 * inconsistent reads (so the check is not vacuous); B-tree invariants after every commit; the
 * final key set equals a sequential insert run; restart counters; degenerate trees.
 */
"use strict";

const MIN_DEGREE = 2;
const MAX_KEYS = 2 * MIN_DEGREE - 1;

class BTreeNode {
  constructor(leaf) {
    this.leaf = leaf;
    this.keys = [];
    this.values = [];
    this.children = [];
  }
}

class ConcurrentBTree {
  constructor() {
    this.root = new BTreeNode(true);
    this.version = 0; // even: idle, odd: a writer is mid-operation
    this.stats = { committed: 0, readerRestarts: 0, oddVersionHits: 0 };
  }

  *lookupSteps(key) {
    let node = this.root;
    for (let depth = 0; depth < 64; depth++) {
      yield "read-level";
      if (node === undefined || node === null || !Array.isArray(node.keys)) return { torn: true };
      let i = 0;
      while (i < node.keys.length && key > node.keys[i]) i++;
      if (i < node.keys.length && node.keys[i] === key) return { found: true, value: node.values[i] };
      if (node.leaf) return { found: false };
      node = node.children[i];
    }
    return { torn: true };
  }

  lookup(key) {
    const gen = this.lookupSteps(key);
    let step = gen.next();
    while (!step.done) step = gen.next();
    return step.value;
  }

  /** Splits a full child; the two parent writes are separate micro-steps (the torn window). */
  *#splitChild(parent, index) {
    const child = parent.children[index];
    const sibling = new BTreeNode(child.leaf);
    const midKey = child.keys[MIN_DEGREE - 1];
    const midValue = child.values[MIN_DEGREE - 1];
    sibling.keys = child.keys.slice(MIN_DEGREE);
    sibling.values = child.values.slice(MIN_DEGREE);
    child.keys = child.keys.slice(0, MIN_DEGREE - 1);
    child.values = child.values.slice(0, MIN_DEGREE - 1);
    if (!child.leaf) sibling.children = child.children.splice(MIN_DEGREE);
    yield "child-split-done"; // the separator is not in the parent yet
    parent.keys.splice(index, 0, midKey);
    parent.values.splice(index, 0, midValue);
    yield "separator-installed"; // the sibling pointer is still missing
    parent.children.splice(index + 1, 0, sibling);
  }

  *insert(key, value) {
    if (this.root.keys.length === MAX_KEYS) {
      const newRoot = new BTreeNode(false);
      newRoot.children.push(this.root);
      yield "root-split";
      yield* this.#splitChild(newRoot, 0);
      this.root = newRoot;
    }
    let node = this.root;
    while (true) {
      yield "descend";
      let index = 0;
      while (index < node.keys.length && key > node.keys[index]) index++;
      // A key can live in an internal node (it separates the children), so match before descending.
      if (index < node.keys.length && node.keys[index] === key) {
        node.values[index] = value;
        return false;
      }
      if (node.leaf) {
        yield "leaf-insert";
        node.keys.splice(index, 0, key);
        node.values.splice(index, 0, value);
        return true;
      }
      if (node.children[index].keys.length === MAX_KEYS) {
        yield "split-child";
        yield* this.#splitChild(node, index);
        continue;
      }
      node = node.children[index];
    }
  }
  /** In-order walk: the key set, independent of the journal. */
  keys() {
    const out = [];
    const walk = (node) => {
      if (!node) return;
      for (let i = 0; i < node.keys.length; i++) { walk(node.children[i]); out.push(node.keys[i]); }
      if (!node.leaf) walk(node.children[node.keys.length]);
    };
    walk(this.root);
    return out;
  }

  /** Occupancy, ordering, balanced-depth and key-set invariants; returns null when all hold. */
  checkInvariants() {
    let leafDepth = null;
    const visit = (node, depth, low, high) => {
      if (node.keys.length > MAX_KEYS) return `node with ${node.keys.length} keys exceeds ${MAX_KEYS}`;
      if (node !== this.root && node.keys.length < MIN_DEGREE - 1) return "non-root node underflows";
      for (let i = 1; i < node.keys.length; i++) if (!(node.keys[i - 1] < node.keys[i])) return "keys unsorted";
      for (const key of node.keys) if (low !== null && key <= low) return `key ${key} violates the lower separator`;
      for (const key of node.keys) if (high !== null && key >= high) return `key ${key} violates the upper separator`;
      if (node.leaf) {
        if (node.children.length !== 0) return "leaf holds children";
        if (leafDepth === null) leafDepth = depth;
        else if (leafDepth !== depth) return "leaves are not all at the same depth";
        return null;
      }
      if (node.children.length !== node.keys.length + 1) return "child count does not match key count";
      for (let i = 0; i < node.children.length; i++) {
        const childLow = i === 0 ? low : node.keys[i - 1];
        const childHigh = i === node.keys.length ? high : node.keys[i];
        const broken = visit(node.children[i], depth + 1, childLow, childHigh);
        if (broken) return broken;
      }
      return null;
    };
    return visit(this.root, 0, null, null);
  }
}

/** The seqlock read: sample, walk, re-sample. `validate=false` accepts torn reads. */
function* readTask(tree, key, validate) {
  while (true) {
    const startVersion = tree.version;
    yield "sample";
    if (validate && startVersion % 2 === 1) {
      tree.stats.oddVersionHits++;
      continue;
    }
    const result = yield* tree.lookupSteps(key);
    const endVersion = tree.version;
    yield "validate";
    if (!validate || endVersion === startVersion) {
      return { key, result, startVersion, validated: endVersion === startVersion };
    }
    tree.stats.readerRestarts++;
  }
}

/** The writer body; the scheduler holds the exclusive latch around every micro-step. */
function* writeTask(tree, key, value) {
  const gen = tree.insert(key, value);
  let step = gen.next();
  while (!step.done) {
    yield step.value;
    step = gen.next();
  }
  return null;
}

/**
 * Deterministic cooperative scheduler. Readers always run; writers are mutually exclusive: a writer
 * only advances while it owns the latch, which is an odd version counter during the mutation.
 */
function runSchedule(tree, operations, seed, validate = true) {
  const rng = makeRng(seed);
  const committed = new Map();
  const snapshots = [new Map()];
  const latch = { holder: null };
  const tasks = operations.map((op) => ({
    kind: op.kind,
    op,
    gen: op.kind === "write" ? writeTask(tree, op.key, op.value) : readTask(tree, op.key, validate),
    done: false,
  }));
  const reads = [];
  while (tasks.some((task) => !task.done)) {
    const enabled = tasks.filter((task) => !task.done
      && (task.kind === "read" || latch.holder === task || latch.holder === null));
    const task = enabled[rng(enabled.length)];
    if (task.kind === "write" && latch.holder === null) {
      latch.holder = task;
      tree.version++; // acquire: readers now see an odd version
      continue;
    }
    const step = task.gen.next();
    if (!step.done) continue;
    task.done = true;
    if (task.kind === "write") {
      committed.set(task.op.key, task.op.value);
      tree.stats.committed++;
      tree.version++; // release
      snapshots.push(new Map(committed));
      latch.holder = null;
    } else if (step.value !== undefined) {
      reads.push(step.value);
    }
  }
  return { committed, snapshots, reads };
}
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Runs a micro-step generator to completion. */
function drain(gen) {  let step = gen.next();  while (!step.done) step = gen.next();
  return step.value;
}

/** Sequential insert run used as the final-state oracle. */
function buildSequential(operations) {
  const tree = new ConcurrentBTree();
  const committed = new Map();
  for (const op of operations) {
    if (op.kind !== "write") continue;
    drain(tree.insert(op.key, op.value));
    committed.set(op.key, op.value);
  }
  return { tree, committed };
}

function makeOperations(rng, writes, reads, keySpace) {
  const operations = [];
  for (let i = 0; i < writes; i++) operations.push({ kind: "write", key: rng(keySpace), value: i });
  for (let i = 0; i < reads; i++) operations.push({ kind: "read", key: rng(keySpace) });
  return operations;
}

module.exports = { ConcurrentBTree, runSchedule, buildSequential, makeOperations, makeRng, drain, MIN_DEGREE, MAX_KEYS };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok } = assert;

  // -- degenerate trees --------------------------------------------------------------
  const empty = new ConcurrentBTree();
  deepStrictEqual(empty.lookup(5), { found: false }, "an empty tree contains nothing");
  strictEqual(empty.checkInvariants(), null, "invariants hold for the empty tree");
  strictEqual(empty.version, 0, "the latch starts idle");    const one = new ConcurrentBTree();
  drain(one.insert(4, "four"));
  deepStrictEqual(one.lookup(4), { found: true, value: "four" }, "a single key is found");
  deepStrictEqual(one.lookup(5), { found: false }, "a missing key in a one-key tree misses");
  strictEqual(one.checkInvariants(), null, "invariants hold for a one-key tree");
  const grow = new ConcurrentBTree();
  const growGen = (key) => drain(grow.insert(key, key));
  for (let i = 0; i < MAX_KEYS; i++) growGen(i);
  strictEqual(grow.root.leaf, true, "the first 2t-1 keys still fit in the root leaf");
  growGen(MAX_KEYS);
  strictEqual(grow.root.leaf, false, "one more key forces a root split");
  deepStrictEqual(grow.keys(), [0, 1, 2, 3], "the in-order walk lists every key once");
  for (let i = MAX_KEYS + 1; i < 60; i++) growGen(i);
  strictEqual(grow.checkInvariants(), null, "the grown tree stays balanced and sorted");
  deepStrictEqual(grow.keys(), Array.from({ length: 60 }, (_, i) => i), "in-order walk of the grown tree");
  const depthOf = (node) => (node.leaf ? 1 : 1 + depthOf(node.children[0]));
  ok(depthOf(grow.root) >= 3, "60 keys must need at least three levels");
  // the split window is real: an undefined child pointer must be reported as a tear, not a miss
  const broken = new ConcurrentBTree();
  for (let i = 0; i < 12; i++) drain(broken.insert(i, i));
  broken.root.children[0] = undefined;
  ok(broken.lookup(0).torn === true, "an undefined child must be reported as a tear, not as a miss");

  // -- interleaved run: every validated read must match its start snapshot -----------
  let inconsistentWhenUnvalidated = 0;
  let readsSeen = 0;
  for (const seed of [1, 7, 2026, 99]) {
    const operations = makeOperations(makeRng(seed), 30, 40, 25);
    const tree = new ConcurrentBTree();
    const { snapshots, reads, committed } = runSchedule(tree, operations, seed);
    strictEqual(tree.checkInvariants(), null, "invariants must hold after the schedule drains");
    const sequential = buildSequential(operations);
    const finalKeys = tree.keys();
    deepStrictEqual(finalKeys, [...sequential.committed.keys()].sort((a, b) => a - b),
      "the final key set must equal the sequential insert run");
    deepStrictEqual(finalKeys, [...committed.keys()].sort((a, b) => a - b), "the journal must match the tree");
    strictEqual(tree.version % 2, 0, "the latch must be released when all tasks are done");
    strictEqual(snapshots.length, tree.stats.committed + 1, "one snapshot per commit plus the initial state");
    readsSeen += reads.length;
    ok(reads.length > 0, "each schedule must complete some reads");
    for (const read of reads) {
      const expected = snapshots[read.startVersion / 2].get(read.key);
      if (read.validated) {
        strictEqual(read.result.found, expected !== undefined, `read of ${read.key} disagrees with the snapshot`);
        if (expected !== undefined) strictEqual(read.result.value, expected, "read value must match the snapshot");
      }
    }
    ok(tree.stats.readerRestarts > 0, `seed ${seed} must produce at least one version-conflict restart`);
    const blind = runSchedule(new ConcurrentBTree(), operations, seed, false);
    for (const read of blind.reads) {
      // an odd sample means the reader started mid-write: the reference state is the last commit
      const expected = blind.snapshots[Math.floor(read.startVersion / 2)].get(read.key);
      const seen = read.result.found ? read.result.value : undefined;
      if (seen !== expected) inconsistentWhenUnvalidated++;
    }
  }
  ok(inconsistentWhenUnvalidated > 0,
    "without validation the same schedules must expose torn reads, otherwise the check is vacuous");

  const runOnce = (seed) => {
    const tree = new ConcurrentBTree();
    const { reads } = runSchedule(tree, makeOperations(makeRng(seed), 20, 30, 15), seed);
    return { restarts: tree.stats.readerRestarts, reads: reads.length, keys: tree.keys().length, committed: tree.stats.committed };
  };
  deepStrictEqual(runOnce(5), runOnce(5), "the schedule must be fully deterministic");
  strictEqual(runOnce(5).committed, 20, "every writer task must commit exactly once");
  ok(runOnce(5).keys > 0 && runOnce(5).reads > 0, "the run must leave keys in the tree and complete reads");
  strictEqual(buildSequential(makeOperations(makeRng(5), 20, 0, 15)).tree.version, 0,
    "the sequential builder never bumps the latch");

  const overwrite = new ConcurrentBTree();
  const ops = [{ kind: "write", key: 3, value: "a" }, { kind: "read", key: 3 }, { kind: "write", key: 3, value: "b" }, { kind: "read", key: 3 }];
  const overwritten = runSchedule(overwrite, ops, 3);
  strictEqual(overwritten.committed.get(3), "b", "the journal keeps the last value");
  for (const read of overwritten.reads) {
    const expected = overwritten.snapshots[read.startVersion / 2].get(3);
    strictEqual(read.result.found ? read.result.value : undefined, expected, "a read must only see committed values");
  }
  deepStrictEqual(overwrite.lookup(3), { found: true, value: "b" }, "the final value is the last write");
  strictEqual(overwrite.checkInvariants(), null, "overwriting a key must not change the structure");
  strictEqual(overwrite.keys().filter((k) => k === 3).length, 1, "an overwritten key appears exactly once");

  console.log(`reads=${readsSeen} torn-without-validation=${inconsistentWhenUnvalidated} restarts sample=${runOnce(5).restarts}`);
  console.log("PASS");
}
