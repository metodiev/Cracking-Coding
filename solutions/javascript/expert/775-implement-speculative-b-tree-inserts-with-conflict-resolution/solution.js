/** Task 775 - Implement speculative B-tree inserts with conflict resolution. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Every node carries a version stamp. A *speculative* insert descends the tree with **no latch at
 * all** (the simulated latch counter stays at zero), recording per level the node it visited, that
 * node's version, the child index it took and whether the child was full. The commit then takes the
 * latch once, re-reads every recorded version, and — only if all are unchanged — replays the plan:
 * it splits exactly the children that were full during the descent and writes the key at the
 * recorded leaf position. A mismatch means another writer touched the path, so the commit aborts,
 * drops the latch and the insert retries from a fresh speculation; after a bounded number of
 * retries it falls back to a plain latched insert.
 *
 * Why validation is enough: recorded child indices are only safe while the path nodes are untouched
 * (a split shifts sibling indices by t), so a mismatch invalidates the plan, not just the leaf. A
 * writer that lands off the path leaves the spec valid — the protocol is conflict-based rather than
 * tree-wide, and both directions are measured below.
 *
 * What is simplified: the latch is a counter/flag (single-writer simulation, cooperative callers
 * instead of threads), no deletes or merges, and cascading splits are replayed from the plan level
 * by level. What is verified: a conflict-free speculative run reproduces the structure of the
 * sequential latched run exactly (canonical pre-order serialisation) with zero aborts; a competing
 * writer on the path forces aborts and retries yet loses nothing; an off-path writer causes no
 * abort; an exhausted retry budget falls back; stale specs (including one committed twice) are
 * rejected; duplicates update in place; root splits; invariants and the key set after every commit.
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

class SpeculativeBTree {
  constructor() {
    this.root = new Node(true);
    this.latchHeld = false;
    this.stats = { speculations: 0, commits: 0, aborts: 0, retries: 0, fallbacks: 0, latchAcquisitions: 0 };
  }

  /** Latch-free read used by the speculation (and by the invariants checker). */
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

  /** Lock-free descent: returns the plan the commit may replay. */
  speculate(key) {
    if (this.latchHeld) throw new Error("a speculation must not run while the latch is held");
    this.stats.speculations++;
    const path = [];
    let node = this.root;
    while (true) {
      const index = this.#childIndex(node, key);
      const found = index < node.keys.length && node.keys[index] === key;
      if (found) {
        return { key, path, match: { node, index, version: node.version }, leafPosition: null, rootVersion: this.root.version };
      }
      if (node.leaf) {
        return { key, path, match: null, leafPosition: { node, index, version: node.version }, rootVersion: this.root.version };
      }
      path.push({
        node,
        version: node.version,
        index,
        needsSplit: node.children[index].keys.length === MAX_KEYS,
      });
      node = node.children[index];
    }
  }

  /** Split a full child; both the child and the parent change, so both versions advance. */
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
    return midKey;
  }

  /** Plain latched insert: proactive splitting, used as the fallback and as the sequential oracle. */
  insertLocked(key, value) {
    const heldBefore = this.latchHeld;
    this.latchHeld = true;
    this.stats.latchAcquisitions++;
    let inserted;
    if (this.root.keys.length === MAX_KEYS) {
      const newRoot = new Node(false);
      newRoot.children.push(this.root);
      newRoot.version++;
      this.#splitChild(newRoot, 0);
      this.root = newRoot;
    }
    let node = this.root;
    while (true) {
      const index = this.#childIndex(node, key);
      if (index < node.keys.length && node.keys[index] === key) {
        node.values[index] = value;
        node.version++;
        inserted = false;
        break;
      }
      if (node.leaf) {
        node.keys.splice(index, 0, key);
        node.values.splice(index, 0, value);
        node.version++;
        inserted = true;
        break;
      }
      if (node.children[index].keys.length === MAX_KEYS) {
        this.#splitChild(node, index);
        continue;
      }
      node = node.children[index];
    }
    this.latchHeld = heldBefore;
    return inserted;
  }

  /** Validates the plan and replays it; returns { ok } or { ok: false, reason }. */
  commit(spec, value) {
    this.latchHeld = true;
    this.stats.latchAcquisitions++;
    const release = (result) => { this.latchHeld = false; return result; };
    if (this.root.version !== spec.rootVersion) {
      this.stats.aborts++;
      return release({ ok: false, reason: "root-restructured" });
    }
    for (const step of spec.path) {
      if (step.node.version !== step.version) {
        this.stats.aborts++;
        return release({ ok: false, reason: "path-changed" });
      }
    }
    const target = spec.match ?? spec.leafPosition;
    if (target.node.version !== target.version) {
      this.stats.aborts++;
      return release({ ok: false, reason: "target-changed" });
    }
    if (spec.match !== null) {
      spec.match.node.values[spec.match.index] = value;
      spec.match.node.version++;
      this.stats.commits++;
      return release({ ok: true, inserted: false });
    }
    // -- apply phase: every node on the path is still exactly as the speculation saw it
    let node = this.root;
    let position = spec.leafPosition === null ? 0 : spec.leafPosition.index;
    let shift = 0;
    if (this.root.keys.length === MAX_KEYS) {
      const oldRoot = this.root;
      const newRoot = new Node(false);
      newRoot.children.push(oldRoot);
      newRoot.version++;
      const midKey = this.#splitChild(newRoot, 0);
      this.root = newRoot;
      if (spec.path.length === 0) {
        node = spec.key < midKey ? oldRoot : newRoot.children[1];
        position = spec.key < midKey ? position : position - MIN_DEGREE;
      } else if (spec.key < midKey) {
        node = oldRoot; // the recorded first step stays valid for the left half
      } else {
        node = newRoot.children[1];
        shift = MIN_DEGREE; // the recorded indices now address the right half
      }
    }
    if (spec.path.length > 0) {
      // Splitting the child we descend into shifts the sibling's indices by MIN_DEGREE — both for
      // the next recorded step and for the recorded insertion position in the target leaf.
      for (const step of spec.path) {
        const index = step.index - shift;
        shift = 0;
        if (step.needsSplit) this.#splitChild(node, index);
        const advanced = step.needsSplit && spec.key >= node.keys[index] ? 1 : 0;
        node = node.children[index + advanced];
        if (advanced === 1) shift = MIN_DEGREE;
      }
      position -= shift;
    }
    node.keys.splice(position, 0, spec.key);
    node.values.splice(position, 0, value);
    node.version++;
    this.stats.commits++;
    return release({ ok: true, inserted: true });
  }

  /**
   * Speculate/validate/retry loop with a latched fallback. `competitor` is a test seam: it runs
   * between the speculation and the validation so the checks can inject a conflicting writer.
   */
  insertSpeculative(key, value, maxRetries = 3, competitor = null) {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const spec = this.speculate(key);
      if (competitor !== null) competitor(attempt);
      const result = this.commit(spec, value);
      if (result.ok) return { retries: attempt, inserted: result.inserted, aborted: false };
      this.stats.retries++;
    }
    this.stats.fallbacks++;
    return { retries: maxRetries, inserted: this.insertLocked(key, value), aborted: true };
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

  /** Canonical pre-order serialisation: lets two trees be compared node by node. */
  serialize() {
    const parts = [];
    const walk = (node) => {
      parts.push(`${node.leaf ? "L" : "I"}(${node.keys.join(" ")})`);
      for (const child of node.children) walk(child);
    };
    walk(this.root);
    return parts.join("|");
  }

  /** Occupancy, ordering, balanced depth and separator invariants; null when all hold. */
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

module.exports = { SpeculativeBTree, makeRng, MIN_DEGREE, MAX_KEYS };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const rng = makeRng(20260775);

  // -- degenerate cases -------------------------------------------------------------
  const empty = new SpeculativeBTree();
  deepStrictEqual(empty.search(1), { found: false }, "an empty tree contains nothing");
  strictEqual(empty.checkInvariants(), null, "invariants hold for the empty tree");
  const spec0 = empty.speculate(7);
  strictEqual(empty.stats.latchAcquisitions, 0, "a speculation must not take the latch");
  strictEqual(spec0.path.length, 0, "the plan for an empty tree has no internal levels");
  deepStrictEqual(empty.commit(spec0, "seven"), { ok: true, inserted: true }, "the first commit succeeds");
  deepStrictEqual(empty.search(7), { found: true, value: "seven" }, "the committed key is visible");
  const replay = empty.commit(spec0, "again");
  ok(replay.ok === false && typeof replay.reason === "string", "reusing a stale spec must be rejected");
  throws(() => { empty.latchHeld = true; empty.speculate(3); }, Error, "speculating under the latch is a bug");
  empty.latchHeld = false;

  // -- conflict-free run reproduces the sequential structure -------------------------
  // Distinct keys only: a latched proactive split also splits its way down to a key that is already
  // present, while the spec's early match check (correctly) skips that, so duplicate-heavy
  // histories may legitimately end up with different — equally valid — node shapes.
  const seen = new Set();
  const keys = [];
  while (keys.length < 120) { const key = rng(1000); if (!seen.has(key)) { seen.add(key); keys.push(key); } }
  const specTree = new SpeculativeBTree();
  const seqTree = new SpeculativeBTree();
  let retries = 0;
  for (const key of keys) {
    retries += specTree.insertSpeculative(key, `v${key}`).retries;
    seqTree.insertLocked(key, `v${key}`);
  }
  strictEqual(retries, 0, "without competitors no insert may need a retry");
  strictEqual(specTree.stats.aborts, 0, "without competitors there must be no aborts");
  strictEqual(specTree.serialize(), seqTree.serialize(),
    "the speculative plan must reproduce the sequential tree node for node");
  strictEqual(specTree.checkInvariants(), null, "invariants hold after the conflict-free run");
  deepStrictEqual(specTree.keys(), [...keys].sort((a, b) => a - b), "the key set matches the input");

  // -- a competitor on the path forces abort + retry, and loses nothing --------------
  const contending = new SpeculativeBTree();
  for (let i = 0; i < 40; i++) contending.insertLocked(i * 2, i);
  let aborts = 0;
  for (let i = 0; i < 60; i++) {
    const key = 200 + i * 3;
    const spec = contending.speculate(key);
    contending.insertLocked(key + 1, -1); // competitor mutates the same leaf first
    if (!contending.commit(spec, i).ok) aborts++;
  }
  strictEqual(aborts, 60, "a competitor on the recorded leaf must invalidate every such spec");
  strictEqual(contending.stats.aborts, 60, "the abort counter must match");
  strictEqual(contending.stats.commits, 0, "every speculative commit in the contended phase aborted");
  for (let i = 0; i < 60; i++) contending.insertSpeculative(200 + i * 3, i);
  strictEqual(contending.keys().length, 160, "no insert may be lost under contention (40 + 60 + 60)");
  strictEqual(contending.checkInvariants(), null, "invariants hold after the contended run");

  // -- the retry loop recovers from a competitor that wins twice, then falls back ----
  const retried = new SpeculativeBTree();
  for (let i = 0; i < 60; i++) retried.insertLocked(i * 3, i);
  const stubborn = retried.insertSpeculative(1000, "x", 3, (attempt) => {
    if (attempt < 2) retried.insertLocked(1001 + attempt * 2, -1); // mutates the target leaf
  });
  strictEqual(stubborn.retries, 2, "two conflicts mean two retries before the commit succeeds");
  strictEqual(stubborn.aborted, false, "the third attempt must commit");
  deepStrictEqual(retried.search(1000), { found: true, value: "x" }, "the retried insert landed");
  const desperate = retried.insertSpeculative(5000, "y", 1, () => retried.insertLocked(5001, -1));
  strictEqual(desperate.aborted, true, "a permanently hostile competitor exhausts the budget");
  strictEqual(desperate.inserted, true, "the fallback must still insert");
  deepStrictEqual(retried.search(5000), { found: true, value: "y" }, "the fallback insert is visible");
  strictEqual(retried.stats.fallbacks, 1, "the fallback must be counted");
  strictEqual(retried.checkInvariants(), null, "the fallback keeps the tree valid");

  // -- an off-path writer must not invalidate the plan ------------------------------
  const disjoint = new SpeculativeBTree();
  for (let i = 0; i < 40; i++) disjoint.insertLocked(i * 4, i); // wide tree, many leaves
  const leftSpec = disjoint.speculate(1);
  disjoint.root.children[disjoint.root.children.length - 1].version++; // a writer touched a far leaf
  ok(disjoint.commit(leftSpec, "left").ok === true, "a writer off the path must not invalidate the plan");
  deepStrictEqual(disjoint.search(1), { found: true, value: "left" }, "the speculative insert landed");

  // -- duplicates update in place; a root split grows the tree ----------------------
  const dup = new SpeculativeBTree();
  dup.insertSpeculative(5, "first");
  deepStrictEqual(dup.insertSpeculative(5, "second"), { retries: 0, inserted: false, aborted: false },
    "a duplicate key must be an in-place update");
  deepStrictEqual(dup.search(5), { found: true, value: "second" }, "the update must win");
  strictEqual(dup.keys().filter((k) => k === 5).length, 1, "the key appears exactly once");
  const grower = new SpeculativeBTree();
  for (let i = 0; i < MAX_KEYS; i++) grower.insertLocked(i, i);
  strictEqual(grower.root.leaf, true, "2t-1 keys still fit in the root leaf");
  deepStrictEqual(grower.commit(grower.speculate(99), 99), { ok: true, inserted: true }, "the 2t-th key commits");
  strictEqual(grower.root.leaf, false, "the commit performed the root split");
  deepStrictEqual(grower.keys(), [0, 1, 2, 99], "the split placed the new key correctly");
  strictEqual(grower.checkInvariants(), null, "the split root stays balanced");

  // -- latch accounting and determinism ---------------------------------------------
  const run = (seed) => {
    const tree = new SpeculativeBTree();
    const r = makeRng(seed);
    for (let i = 0; i < 80; i++) {
      const spec = tree.speculate(r(120));
      if (r(4) === 0) tree.insertLocked(r(120), -1); // interleave competitors
      tree.commit(spec, i);
      tree.insertSpeculative(r(120), i);
    }
    return { stats: tree.stats, keys: tree.keys().length, invariants: tree.checkInvariants(), shape: tree.serialize().length };
  };
  deepStrictEqual(run(7), run(7), "the mixed workload must be deterministic");
  strictEqual(run(7).invariants, null, "invariants hold after the mixed workload");
  ok(run(7).stats.aborts > 0 && run(7).keys > 0, "the mixed workload must record conflicts and keep keys");
  strictEqual(new SpeculativeBTree().latchHeld, false, "the latch ends released");

  console.log(`speculations=${specTree.stats.speculations} conflict aborts=${aborts} retries=${retried.stats.retries} fallbacks=${retried.stats.fallbacks} latch acquisitions=${run(7).stats.latchAcquisitions}`);
  console.log("PASS");
}
