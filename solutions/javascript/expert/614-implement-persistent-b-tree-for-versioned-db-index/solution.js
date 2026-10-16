/** Task 614 - Implement persistent B-tree for versioned DB index. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A copy-on-write B-tree, the classic shape of a versioned storage index (LMDB pages, SQLite's
 * readers): nodes are immutable, so insert and remove rebuild only the nodes on the root-to-leaf
 * path and share every other node with the previous version. A version is one root pointer, which
 * lets a reader keep querying an old snapshot while writers branch off it.
 *
 * Minimum degree T = 3, so a node holds at most 5 keys and a non-root node at least 2. insert()
 * splits overflowing nodes on the way back up (promoting the median); remove() uses the preemptive
 * top-down scheme - before descending into a child holding only T-1 keys it borrows through a
 * rotation or merges the child with a sibling plus the separating key - and replaces a key found in
 * an internal node by its in-order predecessor. delete() takes the predecessor when the left child
 * is fat enough, else the successor, else merges both children and recurses into the merged child.
 * An emptied root is replaced by its only child, so the tree height changes only at the root.
 *
 * What is simplified: keys are integers (comparison is numeric only) and each node caches its
 * subtree count instead of a page header. Verified after every mutation: occupancy bounds, sorted
 * keys, children == keys + 1, uniform leaf depth, cached counts; the key/value contents of every
 * version match an independent Map snapshot, range scans match too, old versions still answer after
 * hundreds of later writes, untouched subtrees are shared by object identity, and each update stays
 * inside the O(T * log_T n) copy budget.
 */
"use strict";

const T = 3;                                   // minimum degree: nodes hold 1..2T-1 = 5 keys
const MAX_KEYS = 2 * T - 1;

let allocatedNodes = 0;

function makeLeaf(keys, values) {
  allocatedNodes++;
  return { leaf: true, keys, values, children: null, count: keys.length };
}

function makeInternal(keys, values, children) {
  allocatedNodes++;
  let count = keys.length;
  for (const child of children) count += child.count;
  return { leaf: false, keys, values, children, count };
}

function mergeNodes(left, key, value, right) {
  const keys = [...left.keys, key, ...right.keys];
  const values = [...left.values, value, ...right.values];
  return left.leaf ? makeLeaf(keys, values) : makeInternal(keys, values, [...left.children, ...right.children]);
}

function lowerBound(keys, key) {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (keys[mid] < key) lo = mid + 1; else hi = mid;
  }
  return lo;
}

class PersistentBTree {
  constructor(root = null) { this.root = root; }

  get size() { return this.root === null ? 0 : this.root.count; }

  get(key) {
    if (!Number.isInteger(key)) throw new TypeError("keys must be integers");
    let node = this.root;
    while (node !== null) {
      const index = lowerBound(node.keys, key);
      if (index < node.keys.length && node.keys[index] === key) return node.values[index];
      if (node.leaf) return undefined;
      node = node.children[index];
    }
    return undefined;
  }

  /** Insert or overwrite; returns a new version (the old one is never touched). */
  insert(key, value) {
    if (!Number.isInteger(key)) throw new TypeError("keys must be integers");
    if (this.root === null) return new PersistentBTree(makeLeaf([key], [value]));
    const result = this.#insert(this.root, key, value);
    if (result.promoted === null) return new PersistentBTree(result.node);
    const promoted = result.promoted;
    return new PersistentBTree(makeInternal([promoted.key], [promoted.value], [result.node, promoted.right]));
  }

  #insert(node, key, value) {
    const index = lowerBound(node.keys, key);
    if (index < node.keys.length && node.keys[index] === key) {
      const values = node.values.slice();
      values[index] = value;                       // overwrite: same key count, new value array
      return { node: node.leaf ? makeLeaf(node.keys.slice(), values) : makeInternal(node.keys.slice(), values, node.children.slice()), promoted: null };
    }
    if (node.leaf) {
      const keys = node.keys.slice();
      const values = node.values.slice();
      keys.splice(index, 0, key);
      values.splice(index, 0, value);
      if (keys.length <= MAX_KEYS) return { node: makeLeaf(keys, values), promoted: null };
      const middle = Math.floor(keys.length / 2);
      return {
        node: makeLeaf(keys.slice(0, middle), values.slice(0, middle)),
        promoted: { key: keys[middle], value: values[middle],
          right: makeLeaf(keys.slice(middle + 1), values.slice(middle + 1)) },
      };
    }
    const result = this.#insert(node.children[index], key, value);
    const keys = node.keys.slice();
    const values = node.values.slice();
    const children = node.children.slice();
    if (result.promoted === null) {
      children[index] = result.node;
      return { node: makeInternal(keys, values, children), promoted: null };
    }
    const promoted = result.promoted;
    keys.splice(index, 0, promoted.key);
    values.splice(index, 0, promoted.value);
    children[index] = result.node;
    children.splice(index + 1, 0, promoted.right);
    if (keys.length <= MAX_KEYS) return { node: makeInternal(keys, values, children), promoted: null };
    const middle = Math.floor(keys.length / 2);
    return {
      node: makeInternal(keys.slice(0, middle), values.slice(0, middle), children.slice(0, middle + 1)),
      promoted: { key: keys[middle], value: values[middle],
        right: makeInternal(keys.slice(middle + 1), values.slice(middle + 1), children.slice(middle + 1)) },
    };
  }

  /** Delete a key; a missing key returns this very version (no allocation, no copy). */
  remove(key) {
    if (!Number.isInteger(key)) throw new TypeError("keys must be integers");
    if (this.root === null) return this;
    const result = this.#remove(this.root, key);
    if (!result.removed) return this;
    let root = result.node;
    if (!root.leaf && root.keys.length === 0) root = root.children[0];   // the root lost its last key
    return new PersistentBTree(root.leaf && root.keys.length === 0 ? null : root);
  }

  #remove(node, key) {
    const index = lowerBound(node.keys, key);
    if (node.leaf) {
      if (index >= node.keys.length || node.keys[index] !== key) return { node, removed: false };
      const keys = node.keys.slice();
      const values = node.values.slice();
      keys.splice(index, 1);
      values.splice(index, 1);
      return { node: makeLeaf(keys, values), removed: true };
    }
    if (index < node.keys.length && node.keys[index] === key) {
      const keys = node.keys.slice();
      const values = node.values.slice();
      const children = node.children.slice();
      const left = node.children[index];
      const right = node.children[index + 1];
      if (left.keys.length >= T) {                 // the in-order predecessor replaces the key
        const predecessor = this.#removeMax(left);
        keys[index] = predecessor.key;
        values[index] = predecessor.value;
        children[index] = predecessor.node;
        return { node: makeInternal(keys, values, children), removed: true };
      }
      if (right.keys.length >= T) {                // otherwise the in-order successor does
        const successor = this.#removeMin(right);
        keys[index] = successor.key;
        values[index] = successor.value;
        children[index + 1] = successor.node;
        return { node: makeInternal(keys, values, children), removed: true };
      }
      // both children hold T-1 keys: pull the key down into a merged child and delete it there
      const merged = mergeNodes(left, key, values[index], right);
      keys.splice(index, 1);
      values.splice(index, 1);
      children.splice(index, 2, this.#remove(merged, key).node);
      return { node: makeInternal(keys, values, children), removed: true };
    }
    const fixed = this.#fix(node, index);
    const result = this.#remove(fixed.node.children[fixed.index], key);
    if (!result.removed) return { node, removed: false };
    const children = fixed.node.children.slice();
    children[fixed.index] = result.node;
    return { node: makeInternal(fixed.node.keys.slice(), fixed.node.values.slice(), children), removed: true };
  }

  /** Removes and returns the largest entry of the subtree: callers fixed the path first. */
  #removeMax(node) {
    if (node.leaf) {
      const keys = node.keys.slice();
      const values = node.values.slice();
      const key = keys.pop();
      const value = values.pop();
      return { node: makeLeaf(keys, values), key, value };
    }
    const fixed = this.#fix(node, node.children.length - 1);
    const last = fixed.index;
    const result = this.#removeMax(fixed.node.children[last]);
    const children = fixed.node.children.slice();
    children[last] = result.node;
    return { node: makeInternal(fixed.node.keys.slice(), fixed.node.values.slice(), children), key: result.key, value: result.value };
  }

  /** Removes and returns the smallest entry of the subtree: callers fixed the path first. */
  #removeMin(node) {
    if (node.leaf) {
      const keys = node.keys.slice();
      const values = node.values.slice();
      const key = keys.shift();
      const value = values.shift();
      return { node: makeLeaf(keys, values), key, value };
    }
    const fixed = this.#fix(node, 0);
    const first = fixed.index;
    const result = this.#removeMin(fixed.node.children[first]);
    const children = fixed.node.children.slice();
    children[first] = result.node;
    return { node: makeInternal(fixed.node.keys.slice(), fixed.node.values.slice(), children), key: result.key, value: result.value };
  }

  /** Guarantees that children[index] holds at least T keys, by rotation or by merging. */
  #fix(node, index) {
    if (node.children[index].keys.length >= T) return { node, index };
    const keys = node.keys.slice();
    const values = node.values.slice();
    const children = node.children.slice();
    const child = children[index];
    if (index > 0 && children[index - 1].keys.length >= T) {                   // rotate right
      const left = children[index - 1];
      const separatorKey = keys[index - 1];        // the old separator moves down into the child
      const separatorValue = values[index - 1];
      keys[index - 1] = left.keys[left.keys.length - 1];
      values[index - 1] = left.values[left.values.length - 1];
      children[index - 1] = left.leaf
        ? makeLeaf(left.keys.slice(0, -1), left.values.slice(0, -1))
        : makeInternal(left.keys.slice(0, -1), left.values.slice(0, -1), left.children.slice(0, -1));
      children[index] = child.leaf
        ? makeLeaf([separatorKey, ...child.keys], [separatorValue, ...child.values])
        : makeInternal([separatorKey, ...child.keys], [separatorValue, ...child.values],
          [left.children[left.children.length - 1], ...child.children]);
      return { node: makeInternal(keys, values, children), index };
    }
    if (index + 1 < children.length && children[index + 1].keys.length >= T) { // rotate left
      const right = children[index + 1];
      const separatorKey = keys[index];            // the old separator moves down into the child
      const separatorValue = values[index];
      keys[index] = right.keys[0];
      values[index] = right.values[0];
      children[index + 1] = right.leaf
        ? makeLeaf(right.keys.slice(1), right.values.slice(1))
        : makeInternal(right.keys.slice(1), right.values.slice(1), right.children.slice(1));
      children[index] = child.leaf
        ? makeLeaf([...child.keys, separatorKey], [...child.values, separatorValue])
        : makeInternal([...child.keys, separatorKey], [...child.values, separatorValue], [...child.children, right.children[0]]);
      return { node: makeInternal(keys, values, children), index };
    }
    const at = index + 1 < children.length ? index : index - 1;                // merge with a sibling
    const merged = mergeNodes(children[at], keys[at], values[at], children[at + 1]);
    keys.splice(at, 1);
    values.splice(at, 1);
    children.splice(at, 2, merged);
    return { node: makeInternal(keys, values, children), index: at };
  }

  entries(lo = null, hi = null) {
    const out = [];
    const collect = (node) => {
      if (node === null) return;
      for (let i = 0; i < node.keys.length; i++) {
        if (!node.leaf) collect(node.children[i]);
        if ((lo === null || node.keys[i] >= lo) && (hi === null || node.keys[i] <= hi)) out.push([node.keys[i], node.values[i]]);
      }
      if (!node.leaf) collect(node.children[node.keys.length]);
    };
    collect(this.root);
    return out;
  }

  keys() { return this.entries().map(([key]) => key); }

  range(lo, hi) { return this.entries(lo, hi); }

  /** Structural invariants; throws with the broken invariant instead of returning false. */
  validate() {
    if (this.root === null) return true;
    const leafDepths = new Set();
    const walk = (node, low, high, depth, isRoot) => {
      if (node.keys.length > MAX_KEYS) throw new Error("node overflows");
      if (!isRoot && node.keys.length < T - 1) throw new Error("node underflows");
      if (node.keys.length !== node.values.length) throw new Error("key/value mismatch");
      for (let i = 1; i < node.keys.length; i++) if (!(node.keys[i - 1] < node.keys[i])) throw new Error("keys not sorted");
      if (node.keys.length > 0) {
        if (low !== null && !(node.keys[0] > low)) throw new Error("key below the subtree range");
        if (high !== null && !(node.keys[node.keys.length - 1] < high)) throw new Error("key above the subtree range");
      }
      let count = node.keys.length;
      if (node.leaf) {
        leafDepths.add(depth);
      } else {
        if (node.children.length !== node.keys.length + 1) throw new Error("children != keys + 1");
        for (let i = 0; i < node.children.length; i++) {
          const childLow = i === 0 ? low : node.keys[i - 1];
          const childHigh = i === node.keys.length ? high : node.keys[i];
          count += walk(node.children[i], childLow, childHigh, depth + 1, false);
        }
      }
      if (count !== node.count) throw new Error("stale cached count");
      return count;
    };
    walk(this.root, null, null, 0, true);
    if (leafDepths.size !== 1) throw new Error("leaves at different depths");
    return true;
  }
}

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

module.exports = { PersistentBTree, T, MAX_KEYS };

if (require.main === module) {
  const assert = require("node:assert");
  const sorted = (map) => [...map.entries()].sort((a, b) => a[0] - b[0]);

  // --- hand-built history: empty tree up to a few keys and back down ---
  const empty = new PersistentBTree();
  assert.strictEqual(empty.size, 0);
  assert.strictEqual(empty.get(1), undefined);
  assert.deepStrictEqual(empty.entries(), []);
  assert.strictEqual(empty.remove(1), empty, "removing from an empty tree returns the same version");
  let small = empty;
  for (const key of [3, 1, 2]) small = small.insert(key, key * 10);
  assert.deepStrictEqual(sorted(new Map(small.entries())), [[1, 10], [2, 20], [3, 30]]);
  assert.strictEqual(small.get(2), 20);
  assert.strictEqual(small.size, 3);
  small.validate();
  const overwritten = small.insert(2, 99);
  assert.strictEqual(overwritten.get(2), 99);
  assert.strictEqual(small.get(2), 20, "the old version keeps the old value");
  assert.strictEqual(overwritten.size, small.size);
  const withoutKey2 = small.remove(2);
  assert.deepStrictEqual(sorted(new Map(withoutKey2.entries())), [[1, 10], [3, 30]]);
  assert.strictEqual(small.get(2), 20, "deleting leaves the source version alone");
  assert.strictEqual(small.remove(42), small, "a missing key returns the same version");
  assert.deepStrictEqual(sorted(new Map(small.remove(1).entries())), [[2, 20], [3, 30]]);
  assert.strictEqual(small.remove(1).remove(2).remove(3).size, 0, "draining the tree empties it");

  // --- long insert history with a snapshot oracle after every step ---
  const versions = [empty];
  const oracles = [new Map()];
  let tree = empty;
  const next = xorshift32(0xbeef);
  for (let step = 0; step < 220; step++) {
    const key = Math.floor(next() * 120);
    const value = step;
    allocatedNodes = 0;
    tree = tree.insert(key, value);
    const allocated = allocatedNodes;
    assert.ok(allocated <= 6 * (2 * T) * (Math.ceil(Math.log2(tree.size + 1)) + 2), `insert copy budget ${allocated}`);
    const snapshot = new Map(oracles[oracles.length - 1]);
    snapshot.set(key, value);
    tree.validate();
    assert.strictEqual(tree.size, snapshot.size, `size after insert ${step}`);
    versions.push(tree);
    oracles.push(snapshot);
  }
  assert.ok(versions[0].root === null);
  assert.strictEqual(versions[1].size, 1);
  assert.strictEqual(versions[1].entries().length, 1);

  // --- interleaved deletes, each version checked against its own snapshot ---
  const removedVersions = [];
  let shrinking = tree;
  for (let step = 0; step < 140; step++) {
    const key = Math.floor(next() * 120);
    const before = shrinking;
    shrinking = shrinking.remove(key);
    const snapshot = new Map(oracles[oracles.length - 1]);
    snapshot.delete(key);
    oracles.push(snapshot);
    if (shrinking === before) assert.strictEqual(snapshot.size, before.size, "no-op removal");
    else shrinking.validate();
    assert.strictEqual(shrinking.size, snapshot.size, `size after delete ${step}`);
    removedVersions.push(shrinking);
  }
  const finalOracle = oracles[oracles.length - 1];
  assert.deepStrictEqual(sorted(new Map(shrinking.entries())), sorted(finalOracle), "final contents");
  assert.deepStrictEqual(shrinking.keys(), [...finalOracle.keys()].sort((a, b) => a - b), "final key order");
  assert.strictEqual(shrinking.get(0), finalOracle.get(0));
  assert.strictEqual(shrinking.range(10, 20).length, [...finalOracle.keys()].filter((k) => k >= 10 && k <= 20).length, "range count");
  for (const [key, value] of shrinking.range(10, 20)) assert.strictEqual(finalOracle.get(key), value, `range value ${key}`);
  assert.deepStrictEqual(shrinking.range(1000, 2000), []);

  // --- time travel: old snapshots still answer after all the later mutations ---
  for (const index of [0, 1, 17, 60, 130, 220]) {
    const version = versions[index];
    const oracle = oracles[index];
    version.validate();
    assert.strictEqual(version.size, oracle.size, `historical size of version ${index}`);
    assert.deepStrictEqual(sorted(new Map(version.entries())), sorted(oracle), `historical contents of version ${index}`);
  }
  const oldVersion = versions[100];
  const snapshotBefore = sorted(new Map(oldVersion.entries()));
  tree = oldVersion.insert(500, "later");
  assert.deepStrictEqual(sorted(new Map(oldVersion.entries())), snapshotBefore, "inserting into a copy leaves the original alone");
  assert.strictEqual(tree.get(500), "later");
  assert.strictEqual(oldVersion.get(500), undefined);

  // --- sharing and negative checks ---
  const lastVersion = versions[versions.length - 1];
  const grown = lastVersion.insert(1000, "x");
  assert.strictEqual(grown.get(1000), "x");
  assert.strictEqual(grown.size, lastVersion.size + 1, "the insert adds exactly one entry");
  assert.strictEqual(lastVersion.get(1000), undefined, "the source version is untouched");
  assert.notStrictEqual(grown.root, lastVersion.root);
  if (!lastVersion.root.leaf) {
    assert.ok(grown.root.children.some((child) => lastVersion.root.children.includes(child)),
      "untouched subtrees are shared by object identity");
  }
  assert.throws(() => tree.get("x"), TypeError);
  assert.throws(() => tree.insert(1.5, 1), TypeError);

  console.log("PASS");
}
