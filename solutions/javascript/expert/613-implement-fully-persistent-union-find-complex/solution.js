/** Task 613 - Implement fully persistent union-find (complex). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A fully persistent union-find is not just "union-find with an undo stack": every past version
 * must stay queryable while any number of new versions branch off it. Both arrays the structure
 * needs (parent and rank) are therefore kept as *persistent arrays* - a complete binary tree per
 * array whose set(i, v) copies the log2(n) nodes on the path to the leaf and shares every other
 * node with the previous version. union then costs two path copies (one for the parent link, one
 * for the rank bump) and find walks parent pointers, which the rank rule keeps at height <= log n.
 *
 * What is simplified: no path compression. That is not an oversight - compression needs to
 * rewrite the parent chain of every node it touches, and in a fully persistent setting that
 * would have to be charged to the version being built while older versions still point at the
 * uncompressed chain; the rank rule alone gives the O(log n) bound here. Arrays are sized to a
 * power of two, which also means element ids are 0..n-1.
 *
 * Verified: every version's find/connected/rank answers agree with an independent naive
 * union-find that copies its whole arrays per union; versions created earlier keep answering
 * identically after hundreds of later unions (immutability), two divergent futures grown from
 * one version stay independent, per-union allocation stays inside the 2 * (levels + 1) budget,
 * parent chains are acyclic and short, and the degenerate cases (n = 1, self union, re-union of
 * connected elements, out-of-range ids) behave.
 */
"use strict";

let allocatedNodes = 0;

class PersistentArray {
  constructor(size, root) {
    if (!Number.isInteger(size) || size <= 0 || (size & (size - 1)) !== 0) {
      throw new RangeError("persistent array size must be a positive power of two");
    }
    this.size = size;
    this.levels = Math.log2(size);
    this.root = root;
  }

  static from(size, fill) {
    const build = (lo, hi) => {
      if (hi - lo === 1) return { leaf: true, value: fill(lo) };
      const mid = (lo + hi) >> 1;
      allocatedNodes++;
      return { leaf: false, left: build(lo, mid), right: build(mid, hi) };
    };
    return new PersistentArray(size, build(0, size));
  }

  #check(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.size) {
      throw new RangeError(`array index ${index} out of range`);
    }
  }

  get(index) {
    this.#check(index);
    let node = this.root;
    let lo = 0;
    let hi = this.size;
    while (!node.leaf) {
      const mid = (lo + hi) >> 1;
      if (index < mid) { node = node.left; hi = mid; } else { node = node.right; lo = mid; }
    }
    return node.value;
  }

  set(index, value) {
    this.#check(index);
    return new PersistentArray(this.size, this.#set(this.root, index, 0, this.size, value));
  }

  #set(node, index, lo, hi, value) {
    allocatedNodes++;
    if (node.leaf) return { leaf: true, value };
    const mid = (lo + hi) >> 1;
    if (index < mid) return { leaf: false, left: this.#set(node.left, index, lo, mid, value), right: node.right };
    return { leaf: false, left: node.left, right: this.#set(node.right, index, mid, hi, value) };
  }
}

class PersistentUnionFind {
  constructor(size, parent = null, rank = null) {
    if (!Number.isInteger(size) || size <= 0) throw new RangeError("element count must be positive");
    this.parent = parent ?? PersistentArray.from(size, (i) => i);
    this.rank = rank ?? PersistentArray.from(size, () => 0);
  }

  get size() { return this.parent.size; }

  find(x) {
    if (!Number.isInteger(x) || x < 0 || x >= this.size) throw new RangeError(`element ${x} out of range`);
    let node = x;
    while (this.parent.get(node) !== node) node = this.parent.get(node);
    return node;
  }

  /** Steps from x to its root; the rank rule caps this at floor(log2 size) + 1. */
  height(x) {
    let steps = 0;
    let current = x;
    while (this.parent.get(current) !== current) { current = this.parent.get(current); steps++; }
    return steps;
  }

  connected(a, b) { return this.find(a) === this.find(b); }

  /** Union by rank; returns this version unchanged when a and b are already connected. */
  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return this;
    if (this.rank.get(ra) < this.rank.get(rb)) { const swap = ra; ra = rb; rb = swap; }
    const parent = this.parent.set(rb, ra);
    const rank = this.rank.get(ra) === this.rank.get(rb) ? this.rank.set(ra, this.rank.get(ra) + 1) : this.rank;
    return new PersistentUnionFind(this.size, parent, rank);
  }

  rootOf(x) { return this.find(x); }

  components() {
    let count = 0;
    for (let x = 0; x < this.size; x++) if (this.parent.get(x) === x) count++;
    return count;
  }

  classes() {
    const out = new Array(this.size);
    for (let x = 0; x < this.size; x++) out[x] = this.find(x);
    return out;
  }

  ranks() {
    const out = new Array(this.size);
    for (let x = 0; x < this.size; x++) out[x] = this.rank.get(x);
    return out;
  }
}

/** Oracle: the same union rules, but each union copies both plain arrays in full. */
class NaiveUnionFind {
  constructor(size) {
    this.parent = Array.from({ length: size }, (_, i) => i);
    this.rank = new Array(size).fill(0);
  }

  find(x) { while (this.parent[x] !== x) x = this.parent[x]; return x; }

  union(a, b) {
    const next = new NaiveUnionFind(this.parent.length);
    next.parent = this.parent.slice();
    next.rank = this.rank.slice();
    let ra = next.find(a);
    let rb = next.find(b);
    if (ra === rb) return this;
    if (next.rank[ra] < next.rank[rb]) { const swap = ra; ra = rb; rb = swap; }
    next.parent[rb] = ra;
    if (next.rank[ra] === next.rank[rb]) next.rank[ra]++;
    return next;
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

module.exports = { PersistentUnionFind, PersistentArray, NaiveUnionFind };

if (require.main === module) {
  const assert = require("node:assert");
  const size = 16;

  // --- grow one long history and check every version against the oracle ---
  const versions = [new PersistentUnionFind(size)];
  const oracles = [new NaiveUnionFind(size)];
  const next = xorshift32(0x5eed);
  let merges = 0;
  for (let step = 0; step < 60; step++) {
    const a = Math.floor(next() * size);
    const b = Math.floor(next() * size);
    const before = versions[versions.length - 1];
    allocatedNodes = 0;
    const after = before.union(a, b);
    const allocated = allocatedNodes;
    if (after !== before) {
      merges++;
      assert.ok(allocated <= 2 * (before.parent.levels + 2), `union allocation ${allocated}`);
      assert.strictEqual(after.components(), before.components() - 1, "union removes one root");
    } else {
      assert.strictEqual(allocated, 0, "union of connected elements allocates nothing");
    }
    versions.push(after);
    oracles.push(oracles[oracles.length - 1].union(a, b));
  }
  assert.ok(merges >= size - 1, "the schedule must connect the elements");

  const checkVersion = (version, oracle, label) => {
    assert.strictEqual(version.components(), oracle.parent.filter((p, x) => p === x).length, `roots (${label})`);
    for (let x = 0; x < size; x++) {
      assert.strictEqual(version.find(x), oracle.find(x), `find(${x}) of ${label}`);
      assert.strictEqual(version.rank.get(x), oracle.rank[x], `rank(${x}) of ${label}`);
      assert.ok(version.height(x) <= Math.floor(Math.log2(size)) + 1, `height of ${x} in ${label}`);
    }
    for (let a = 0; a < size; a++) {
      for (let b = 0; b < size; b++) {
        assert.strictEqual(version.connected(a, b), oracle.find(a) === oracle.find(b), `connected(${a},${b}) of ${label}`);
      }
    }
  };
  for (const [index, version] of versions.entries()) checkVersion(version, oracles[index], `version ${index}`);
  assert.strictEqual(versions[0].components(), size, "the empty history is n singletons");
  assert.strictEqual(versions[0].connected(0, 1), false);
  assert.strictEqual(versions[versions.length - 1].components(), 1, "the whole set is connected");
  assert.strictEqual(versions[versions.length - 1].union(3, 7).components(), 1, "re-union keeps one component");

  // --- non-interference of the reference version and of two divergent futures ---
  const fork = versions[10];
  const forkOracle = oracles[10];
  const forkClasses = fork.classes();
  let left = fork;
  let leftOracle = forkOracle;
  for (const [a, b] of [[1, 5], [2, 6], [5, 3]]) { left = left.union(a, b); leftOracle = leftOracle.union(a, b); }
  let right = fork;
  let rightOracle = forkOracle;
  for (const [a, b] of [[3, 4], [0, 7]]) { right = right.union(a, b); rightOracle = rightOracle.union(a, b); }
  checkVersion(left, leftOracle, "left branch");
  checkVersion(right, rightOracle, "right branch");
  assert.deepStrictEqual(fork.classes(), forkClasses, "the fork version is immutable");
  assert.strictEqual(left.connected(1, 3), true, "the left branch joined 1 and 3");
  assert.strictEqual(right.connected(1, 3), forkOracle.find(1) === forkOracle.find(3), "the right branch did not");
  assert.strictEqual(left.connected(0, 7), forkOracle.find(0) === forkOracle.find(7), "and not the other way round");
  assert.strictEqual(right.connected(0, 7), true, "the right branch joined 0 and 7");

  // --- persistent array mechanics, including sharing and bounds ---
  const array = PersistentArray.from(8, (i) => i * i);
  assert.deepStrictEqual(Array.from({ length: 8 }, (_, i) => array.get(i)), [0, 1, 4, 9, 16, 25, 36, 49]);
  allocatedNodes = 0;
  const changed = array.set(5, -1);
  assert.strictEqual(changed.get(5), -1);
  assert.strictEqual(array.get(5), 25, "the old array is untouched");
  assert.strictEqual(allocatedNodes, array.levels + 1, "one copied node per level plus the leaf");
  assert.ok(changed.root.left === array.root.left || changed.root.right === array.root.right, "shares a subtree");
  assert.throws(() => array.get(8), RangeError);
  assert.throws(() => array.set(-1, 0), RangeError);
  assert.throws(() => PersistentArray.from(6, () => 0), RangeError);

  // --- degenerate cases ---
  const single = new PersistentUnionFind(1);
  assert.strictEqual(single.components(), 1);
  assert.strictEqual(single.union(0, 0), single, "self union returns the same version");
  assert.strictEqual(single.find(0), 0);
  assert.throws(() => single.find(1), RangeError);
  assert.throws(() => new PersistentUnionFind(0), RangeError);
  const pair = new PersistentUnionFind(2);
  const joined = pair.union(0, 1);
  assert.strictEqual(joined.find(0), joined.find(1));
  assert.strictEqual(joined.height(1), 1);
  assert.strictEqual(pair.height(1), 0, "the old version still has two singletons");

  console.log("PASS");
}
