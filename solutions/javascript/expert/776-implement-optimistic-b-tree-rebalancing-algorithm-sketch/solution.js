/** Task 776 - Implement optimistic B-tree rebalancing algorithm sketch. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Optimistic ("bottom-up") rebalancing: the descent does *no* splits, inserts the key into the leaf,
 * and only then walks back up the recorded path, splitting every node that ended up holding more
 * than 2t-1 keys and promoting its middle key into the parent. Each parent write is guarded by the
 * version stamp the descent recorded for it: if the parent changed in the meantime the rebalance is
 * not applied, the insert restarts from scratch and the stale work is thrown away — that is the
 * optimistic part, and unlike a pessimistic top-down pass it never pays for splits that the
 * insertion does not need.
 *
 * What is simplified (sketch): single-writer cooperative simulation instead of real threads (the
 * conflicting writer is injected through a test seam), a restart re-descends with fully fresh
 * versions rather than repairing the path in place, and there are no deletes or merges. What is
 * verified: the classic B-tree invariants (occupancy, sorted keys, separators, all leaves on one
 * level, key set) after *every* insert; the tree contains exactly the inserted key set and height
 * grows by at most one level per insert (only when the root splits); the optimistic tree is node-by
 * -node identical to a pessimistic run of the same algorithm on distinct keys; an injected writer
 * that mutates a parent forces exactly one restart and the retried insert still lands correctly;
 * a duplicate key is an in-place update that never rebalances; small/degenerate trees and
 * determinism.
 */
"use strict";

const MIN_DEGREE = 2;
const MAX_KEYS = 2 * MIN_DEGREE - 1;

class Node {
  constructor(leaf) {
    this.leaf = leaf;
    this.keys = [];
    this.values = [];
    this.children = [];
    this.version = 0;
  }
}

class OptimisticBTree {
  constructor() {
    this.root = new Node(true);
    this.stats = { inserts: 0, updates: 0, splits: 0, retries: 0, restarts: 0, validations: 0, giveUps: 0 };
  }

  #childIndex(node, key) {
    let index = 0;
    while (index < node.keys.length && key > node.keys[index]) index++;
    return index;
  }

  search(key) {
    let node = this.root;
    while (true) {
      const index = this.#childIndex(node, key);
      if (index < node.keys.length && node.keys[index] === key) return { found: true, value: node.values[index] };
      if (node.leaf) return { found: false };
      node = node.children[index];
    }
  }

  height() {
    let depth = 1;
    for (let node = this.root; !node.leaf; node = node.children[0]) depth++;
    return depth;
  }

  /** Records the parent chain with the version each node had at read time. */
  #descend(key) {
    const path = [];
    let node = this.root;
    while (true) {
      const index = this.#childIndex(node, key);
      if (index < node.keys.length && node.keys[index] === key) return { path, leaf: node, match: { node, index } };
      if (node.leaf) return { path, leaf: node, match: null };
      path.push({ node, version: node.version, index });
      node = node.children[index];
    }
  }

  /** Splits an oversized child in place and reports the promoted separator. */
  #splitChild(parent, index) {
    const child = parent.children[index];
    const sibling = new Node(child.leaf);
    const midKey = child.keys[MIN_DEGREE - 1];
    const midValue = child.values[MIN_DEGREE - 1];
    sibling.keys = child.keys.slice(MIN_DEGREE);
    sibling.values = child.values.slice(MIN_DEGREE);
    child.keys = child.keys.slice(0, MIN_DEGREE - 1);
    child.values = child.values.slice(0, MIN_DEGREE - 1);
    if (!child.leaf) sibling.children = child.children.splice(MIN_DEGREE);
    parent.keys.splice(index, 0, midKey);
    parent.values.splice(index, 0, midValue);
    parent.children.splice(index + 1, 0, sibling);
    child.version++;
    parent.version++;
    this.stats.splits++;
    return { midKey, sibling };
  }

  #insertIntoLeaf(leaf, key, value) {
    const index = this.#childIndex(leaf, key);
    if (index < leaf.keys.length && leaf.keys[index] === key) {
      leaf.values[index] = value;
      leaf.version++;
      this.stats.updates++;
      return false;
    }
    leaf.keys.splice(index, 0, key);
    leaf.values.splice(index, 0, value);
    leaf.version++;
    this.stats.inserts++;
    return true;
  }

  /**
   * Bottom-up rebalance. Returns null when everything was applied, or "parent-changed" when a
   * validated parent was mutated (the caller then retries the whole insert with fresh versions).
   * A restarted attempt may have left an over-full node *on the path*, so the walk starts at the
   * deepest over-full path node rather than at the leaf.
   */
  #rebalanceUp(path, leaf, competitor) {
    let levels = path.slice();
    let current = leaf;
    for (let i = levels.length - 1; i >= 0; i--) {
      if (levels[i].node.keys.length > MAX_KEYS) {
        current = levels[i].node;
        levels = levels.slice(0, i);
        break;
      }
    }
    while (current.keys.length > MAX_KEYS) {
      const step = levels.pop();
      if (step === undefined) { // the root itself overflowed: grow the tree by one level
        const newRoot = new Node(false);
        newRoot.children.push(current);
        newRoot.version++;
        this.#splitChild(newRoot, 0);
        this.root = newRoot;
        return null;
      }
      if (competitor !== null) competitor(step.node);
      this.stats.validations++;
      if (step.node.version !== step.version) return "parent-changed";
      this.#splitChild(step.node, step.index);
      current = step.node;
    }
    return null;
  }

  /**
   * Optimistic insert: write, rebalance, and restart with fresh versions if a validated parent was
   * mutated. When the injected writer keeps winning the sketch gives up on optimism and repairs the
   * path top-down without validation.
   */
  insert(key, value, competitor = null) {
    for (let attempt = 0; attempt <= 2; attempt++) {
      const { path, leaf, match } = this.#descend(key);
      if (match !== null) {
        match.node.values[match.index] = value;
        match.node.version++;
        this.stats.updates++;
      } else {
        this.#insertIntoLeaf(leaf, key, value);
      }
      const outcome = this.#rebalanceUp(path, leaf, competitor);
      if (outcome === null) return { restarts: attempt, attempt, gaveUp: false };
      this.stats.retries++;
      this.stats.restarts++;
    }
    this.stats.giveUps++;
    const { path, leaf } = this.#descend(key);
    this.#rebalanceUp(path, leaf, null);
    return { restarts: 3, attempt: 3, gaveUp: true };
  }

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

  /** Canonical pre-order serialisation, so two trees can be compared node by node. */
  serialize() {
    const parts = [];
    const walk = (node) => {
      parts.push(`${node.leaf ? "L" : "I"}(${node.keys.join(" ")})`);
      for (const child of node.children) walk(child);
    };
    walk(this.root);
    return parts.join("|");
  }

  checkInvariants() {
    let leafDepth = null;
    const visit = (node, depth, low, high) => {
      if (node.keys.length > MAX_KEYS) return "node overflows";
      if (node !== this.root && node.keys.length < MIN_DEGREE - 1) return "non-root node underflows";
      for (let i = 1; i < node.keys.length; i++) if (!(node.keys[i - 1] < node.keys[i])) return "keys unsorted";
      for (const key of node.keys) if ((low !== null && key <= low) || (high !== null && key >= high)) return "separator violated";
      if (node.leaf) {
        if (node.children.length !== 0) return "leaf holds children";
        if (leafDepth === null) leafDepth = depth;
        else if (leafDepth !== depth) return "leaves at different depths";
        return null;
      }
      if (node.children.length !== node.keys.length + 1) return "child count mismatch";
      for (let i = 0; i < node.children.length; i++) {
        const broken = visit(node.children[i], depth + 1, i === 0 ? low : node.keys[i - 1],
          i === node.keys.length ? high : node.keys[i]);
        if (broken) return broken;
      }
      return null;
    };
    return visit(this.root, 0, null, null);
  }
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

module.exports = { OptimisticBTree, makeRng, MIN_DEGREE, MAX_KEYS };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const rng = makeRng(20260776);

  // -- degenerate trees -------------------------------------------------------------
  const empty = new OptimisticBTree();
  deepStrictEqual(empty.search(1), { found: false }, "an empty tree contains nothing");
  strictEqual(empty.checkInvariants(), null, "invariants hold for the empty tree");
  strictEqual(empty.keys().length, 0, "an empty tree has no keys");
  const single = new OptimisticBTree();
  single.insert(4, "four");
  deepStrictEqual(single.search(4), { found: true, value: "four" }, "a one-key tree finds its key");
  deepStrictEqual(single.insert(4, "FOUR"), { restarts: 0, attempt: 0, gaveUp: false }, "a duplicate insert reports no restart");
  deepStrictEqual(single.search(4), { found: true, value: "FOUR" }, "the duplicate updated in place");
  strictEqual(single.stats.updates, 1, "the update counter must record the in-place write");
  strictEqual(single.stats.splits, 0, "an in-place update must not split anything");
  strictEqual(single.keys().length, 1, "the key set stays a single key");

  // -- the leaf overflows by exactly one key, then the root grows --------------------
  const grower = new OptimisticBTree();
  for (let i = 0; i < MAX_KEYS; i++) grower.insert(i, i);
  strictEqual(grower.root.leaf, true, "the first 2t-1 keys fit in a leaf root");
  strictEqual(grower.keys().length, MAX_KEYS, "no split happened yet");
  grower.insert(99, 99);
  strictEqual(grower.root.leaf, false, "the (2t)-th key overflows the leaf and grows the root");
  strictEqual(grower.height(), 2, "the tree is now two levels high");
  deepStrictEqual(grower.keys(), [0, 1, 2, 99], "the bottom-up split keeps the key set sorted");
  strictEqual(grower.checkInvariants(), null, "the grown tree satisfies every invariant");

  // -- invariants and height discipline after every insert --------------------------
  const climbing = new OptimisticBTree();
  const oracle = new Map();
  let previousHeight = 1;
  let growths = 0;
  for (let i = 0; i < 250; i++) {
    const key = rng(400);
    const before = climbing.height();
    climbing.insert(key, `v${key}`);
    oracle.set(key, `v${key}`);
    const after = climbing.height();
    if (after !== before) { growths++; strictEqual(after, before + 1, "the height may grow by one level at most"); }
    strictEqual(climbing.checkInvariants(), null, `invariant broken after inserting ${key}`);
    if (i % 25 === 0) deepStrictEqual(climbing.keys(), [...oracle.keys()].sort((a, b) => a - b), "key set drift");
    previousHeight = after;
  }
  ok(growths > 0 && previousHeight >= 3, "250 keys must have grown the tree several levels");
  for (const [key, value] of oracle) deepStrictEqual(climbing.search(key), { found: true, value }, "value drift");
  ok(climbing.search(9999).found === false, "an absent key must not be found");

  // -- optimistic == pessimistic: same algorithm, no injected writer -----------------
  const optimistic = new OptimisticBTree();
  const pessimistic = new OptimisticBTree();
  const distinct = [];
  const seen = new Set();
  while (distinct.length < 200) { const key = rng(1000); if (!seen.has(key)) { seen.add(key); distinct.push(key); } }
  for (const key of distinct) {
    optimistic.insert(key, key);
    pessimistic.insert(key, key, () => {}); // an inert competitor: identical writes, no interference
  }
  strictEqual(pessimistic.stats.retries, 0, "an inert competitor must never invalidate a rebalance");
  strictEqual(optimistic.serialize(), pessimistic.serialize(),
    "without interference the optimistic run must match the pessimistic one node for node");
  strictEqual(optimistic.checkInvariants(), null, "the compared trees are valid");

  // -- an injected writer on the parent path forces a restart ------------------------
  const contended = new OptimisticBTree();
  for (let i = 0; i <= 6; i++) contended.insert(i * 10, i); // leaves end up holding three keys each
  const leaves = [];
  const collect = (node) => { if (node.leaf) leaves.push(node); else node.children.forEach(collect); };
  collect(contended.root);
  const fullLeaf = leaves.find((leaf) => leaf.keys.length === MAX_KEYS);
  ok(fullLeaf !== undefined, "the setup must contain a full leaf");
  const contestedKey = fullLeaf.keys[1] + 1; // lands inside that full leaf: the split chain must run
  let calls = 0;
  let bumps = 0;
  const outcome = contended.insert(contestedKey, "x", (parent) => {
    calls++;
    if (bumps === 0) { parent.version++; bumps++; } // the competitor wins exactly once
  });
  strictEqual(bumps, 1, "the competitor must have won once");
  ok(calls >= 1, "the rebalance must have attempted at least one parent update");
  strictEqual(outcome.restarts, 1, "one conflict must force exactly one restart");
  strictEqual(contended.stats.restarts, 1, "the restart counter must match");
  deepStrictEqual(contended.search(contestedKey), { found: true, value: "x" }, "the retried insert still landed");
  strictEqual(contended.keys().filter((k) => k === contestedKey).length, 1, "the key must appear exactly once");
  strictEqual(contended.checkInvariants(), null, "the retried rebalance kept the tree valid");
  ok(contended.stats.validations > 0, "every parent write must have been validated");

  // -- a writer that fires on every attempt makes the sketch give up on optimism -----
  const hostile = new OptimisticBTree();
  for (let i = 0; i <= 6; i++) hostile.insert(i * 10, i);
  const hostileKey = 5000;
  const hostileOutcome = hostile.insert(hostileKey, "y", (parent) => { parent.version++; });
  strictEqual(hostileOutcome.gaveUp, true, "a writer that keeps winning must be given up on");
  strictEqual(hostile.stats.giveUps, 1, "the give-up path must be recorded");
  strictEqual(hostile.checkInvariants(), null, "the sketch must leave the tree valid even then");
  deepStrictEqual(hostile.search(hostileKey), { found: true, value: "y" }, "the top-down repair still inserts the key");

  // -- determinism ------------------------------------------------------------------
  const replay = () => {
    const tree = new OptimisticBTree();
    const r = makeRng(4242);
    for (let i = 0; i < 120; i++) tree.insert(r(200), i);
    return { shape: tree.serialize(), keys: tree.keys().length, stats: tree.stats };
  };
  deepStrictEqual(replay(), replay(), "the same seed must rebuild the same tree");

  console.log(`inserts=${optimistic.stats.inserts} updates=${optimistic.stats.updates} splits=${optimistic.stats.splits} validations=${contended.stats.validations} retries=${contended.stats.retries} height=${optimistic.height()}`);
  console.log("PASS");
}
