/** Task 612 - Implement advanced persistent data structures with path copying optimization. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Three fully persistent structures on one idea: never mutate a node, copy the O(log n) nodes on
 * the update path and share everything else with the previous version. A version is just the root
 * pointer, so many versions coexist and each keeps answering with its own values.
 *
 *   PersistentStack  - singly linked list, push/pop share the tail, O(1) per version.
 *   PersistentMap    - treap keyed by a deterministic priority hash(key), built only from persistent
 *                      split/merge; insert/remove/get are O(log n) node copies.
 *   PersistentVector - complete binary tree over a power-of-two slot count; set(i, v) copies exactly
 *                      one node per level, the path-copying optimum.
 *
 * What is simplified: integer keys (the double split that drops the overwritten key relies on that),
 * priorities from a fixed hash instead of a random source, power-of-two vector slots. Verified:
 * every version matches an independent plain-data snapshot taken when it was created, old versions
 * are re-read *after* newer updates to prove immutability, allocation counts stay on the O(log n)
 * path-copying budget (compared against a naive full copy), untouched subtrees are shared by object
 * identity, and the treap/vector invariants (BST order, heap order, cached sizes, uniform leaf
 * depth) hold after every change.
 */
"use strict";

let allocationCount = 0;

function resetAllocationCount() { allocationCount = 0; }

class PersistentStack {
  constructor(top = null) { this.top = top; }

  push(value) { return new PersistentStack({ value, next: this.top }); }

  pop() {
    if (this.top === null) throw new Error("pop from an empty stack");
    return { stack: new PersistentStack(this.top.next), value: this.top.value };
  }

  peek() { return this.top === null ? undefined : this.top.value; }

  get size() {
    let count = 0;
    for (let node = this.top; node !== null; node = node.next) count++;
    return count;
  }

  toArray() {
    const out = [];
    for (let node = this.top; node !== null; node = node.next) out.push(node.value);
    return out;
  }
}

/** Deterministic priority: a splitmix-style mix of the key, so the treap shape is fixed. */
function priorityOf(key, seed = 0x9e3779b9) {
  let x = (Math.imul(key, 0x85ebca6b) ^ seed) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

class TreapNode {
  constructor(key, value, priority, left, right) {
    this.key = key;
    this.value = value;
    this.priority = priority;
    this.left = left;
    this.right = right;
    this.size = 1 + (left === null ? 0 : left.size) + (right === null ? 0 : right.size);
    allocationCount++;
  }
}

/** Copies a node with some children/value replaced; the cached size is rebuilt in O(1). */
function copyNode(node, changes) {
  return new TreapNode(
    changes.key ?? node.key,
    changes.value ?? node.value,
    node.priority,
    changes.left !== undefined ? changes.left : node.left,
    changes.right !== undefined ? changes.right : node.right,
  );
}

class PersistentMap {
  constructor(root = null) { this.root = root; }

  get size() { return this.root === null ? 0 : this.root.size; }

  get(key) {
    let node = this.root;
    while (node !== null) {
      if (key === node.key) return node.value;
      node = key < node.key ? node.left : node.right;
    }
    return undefined;
  }

  insert(key, value) {
    const [left, rest] = PersistentMap.#split(this.root, key);
    const [, right] = PersistentMap.#split(rest, key + 1);       // drop the old key == key node
    const fresh = new TreapNode(key, value, priorityOf(key), null, null);
    return new PersistentMap(PersistentMap.#merge(left, PersistentMap.#merge(fresh, right)));
  }

  remove(key) {
    const [left, rest] = PersistentMap.#split(this.root, key);
    const [middle, right] = PersistentMap.#split(rest, key + 1);
    if (middle === null) return this;                            // missing key: same version
    return new PersistentMap(PersistentMap.#merge(left, right));
  }

  static #split(node, key) {
    if (node === null) return [null, null];
    if (key <= node.key) {
      const [left, rest] = PersistentMap.#split(node.left, key);
      return [left, copyNode(node, { left: rest })];
    }
    const [left, rest] = PersistentMap.#split(node.right, key);
    return [copyNode(node, { right: left }), rest];
  }

  static #merge(a, b) {
    if (a === null) return b;
    if (b === null) return a;
    if (a.priority <= b.priority) return copyNode(a, { right: PersistentMap.#merge(a.right, b) });
    return copyNode(b, { left: PersistentMap.#merge(a, b.left) });
  }

  keys() {
    const out = [];
    const walk = (node) => { if (node === null) return; walk(node.left); out.push(node.key); walk(node.right); };
    walk(this.root);
    return out;
  }

  entries() {
    const out = [];
    const walk = (node) => {
      if (node === null) return;
      walk(node.left);
      out.push([node.key, node.value]);
      walk(node.right);
    };
    walk(this.root);
    return out;
  }

  /** Structural invariants: BST order, heap order on priorities, cached sizes. */
  validate() {
    const check = (node, low, high) => {
      if (node === null) return 0;
      if (low !== null && !(node.key > low)) throw new Error(`BST order broken at key ${node.key}`);
      if (high !== null && !(node.key < high)) throw new Error(`BST order broken at key ${node.key}`);
      for (const child of [node.left, node.right]) {
        if (child !== null && child.priority < node.priority) throw new Error("heap order broken");
      }
      const size = 1 + check(node.left, low, node.key) + check(node.right, node.key, high);
      if (size !== node.size) throw new Error("stale cached size");
      return size;
    };
    check(this.root, null, null);
    return true;
  }
}

class VectorNode {
  constructor(left, right) {
    this.leaf = false;
    this.left = left;
    this.right = right;
    allocationCount++;
  }

  static leaf(value, counter) {
    const node = Object.create(VectorNode.prototype);
    node.leaf = true;
    node.value = value;
    node.left = null;
    node.right = null;
    counter.nodes++;
    return node;
  }
}

class PersistentVector {
  constructor(slots, root = null) {
    if (!Number.isInteger(slots) || slots <= 0 || (slots & (slots - 1)) !== 0) {
      throw new RangeError("slot count must be a positive power of two");
    }
    this.slots = slots;
    this.levels = Math.log2(slots);
    this.counter = { nodes: 0 };
    this.allocations = 0;
    this.root = root === null ? this.#build(0) : root;
  }

  #build(level) {
    if (level === this.levels) return VectorNode.leaf(null, this.counter);
    return new VectorNode(this.#build(level + 1), this.#build(level + 1));
  }

  #versionCount(node, level) {
    if (node.leaf) return node.value === null ? 0 : 1;
    return this.#versionCount(node.left, level + 1) + this.#versionCount(node.right, level + 1);
  }

  get liveSlots() { return this.#versionCount(this.root, 0); }

  get(index) {
    if (index < 0 || index >= this.slots) throw new RangeError(`slot ${index} out of range`);
    let node = this.root;
    let low = 0;
    let high = this.slots;
    while (!node.leaf) {
      const middle = (low + high) / 2;
      if (index < middle) { node = node.left; high = middle; } else { node = node.right; low = middle; }
    }
    return node.value;
  }

  /** Copy exactly one node per level: the path-copying optimum for a fixed-depth tree. */
  set(index, value) {
    if (index < 0 || index >= this.slots) throw new RangeError(`slot ${index} out of range`);
    const before = allocationCount;
    const root = this.#set(this.root, index, 0, this.slots, value);
    const next = new PersistentVector(this.slots, root);
    next.counter = this.counter;
    next.allocations = allocationCount - before;                 // one internal node per level
    return next;
  }

  #set(node, index, low, high, value) {
    if (node.leaf) return VectorNode.leaf(value, this.counter);
    const middle = (low + high) / 2;
    if (index < middle) return new VectorNode(this.#set(node.left, index, low, middle, value), node.right);
    return new VectorNode(node.left, this.#set(node.right, index, middle, high, value));
  }

  toArray() {
    const out = [];
    for (let i = 0; i < this.slots; i++) if (this.get(i) !== null) out.push([i, this.get(i)]);
    return out;
  }

  /** Complete tree invariant: every leaf sits at the same depth. */
  validate() {
    const depth = (node) => {
      if (node.leaf) return 0;
      const left = depth(node.left);
      const right = depth(node.right);
      if (left !== right) throw new Error("vector is not a complete tree");
      return left + 1;
    };
    if (depth(this.root) !== this.levels) throw new Error("vector depth mismatch");
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

module.exports = { PersistentStack, PersistentMap, PersistentVector, priorityOf, resetAllocationCount, liveAllocations: () => allocationCount };

if (require.main === module) {
  const assert = require("node:assert");

  // --- persistent stack: every version must keep its own contents ---
  const stackVersions = [new PersistentStack()];
  const stackOracle = [[]];
  for (let i = 0; i < 12; i++) {
    const top = stackVersions[stackVersions.length - 1];
    stackVersions.push(top.push(i * i));
    stackOracle.push([i * i, ...stackOracle[stackOracle.length - 1]]);
  }
  for (const [index, version] of stackVersions.entries()) {
    assert.deepStrictEqual(version.toArray(), stackOracle[index], `stack version ${index}`);
    assert.strictEqual(version.size, stackOracle[index].length, `stack size ${index}`);
  }
  const popped = stackVersions[12].pop();
  assert.strictEqual(popped.value, 121);
  assert.deepStrictEqual(popped.stack.toArray(), stackOracle[11], "pop yields the previous version");
  assert.deepStrictEqual(stackVersions[12].toArray(), stackOracle[12], "pop does not mutate the version");
  assert.strictEqual(new PersistentStack().peek(), undefined);
  assert.throws(() => new PersistentStack().pop(), /empty stack/);

  // --- persistent treap map: snapshots, time travel, overwrite semantics ---
  resetAllocationCount();
  let map = new PersistentMap();
  const mapVersions = [map];
  const mapOracles = [new Map()];
  for (let i = 0; i < 40; i++) {
    const key = (i * 37) % 64;
    map = map.insert(key, i);
    const snapshot = new Map(mapOracles[mapOracles.length - 1]);
    snapshot.set(key, i);
    mapVersions.push(map);
    mapOracles.push(snapshot);
  }
  for (const [index, version] of mapVersions.entries()) {
    version.validate();
    const oracle = mapOracles[index];
    assert.strictEqual(version.size, oracle.size, `map size of version ${index}`);
    assert.deepStrictEqual(version.keys(), [...oracle.keys()].sort((a, b) => a - b), `map keys of version ${index}`);
    for (const [key, value] of oracle) assert.strictEqual(version.get(key), value, `map value ${key} of version ${index}`);
    for (let key = 0; key < 64; key++) {
      if (!oracle.has(key)) assert.strictEqual(version.get(key), undefined, `absent key ${key} in version ${index}`);
    }
  }
  const beforeUpdate = mapVersions[10];
  const beforeSnapshot = beforeUpdate.entries();
  const updated = beforeUpdate.insert(999, "late");
  assert.deepStrictEqual(beforeUpdate.entries(), beforeSnapshot, "an old version is immutable");
  assert.strictEqual(beforeUpdate.get(999), undefined);
  assert.strictEqual(updated.get(999), "late");
  assert.notStrictEqual(updated.root, beforeUpdate.root);
  assert.ok(updated.root.left === beforeUpdate.root.left || updated.root.right === beforeUpdate.root.right,
    "a new version must share at least one untouched subtree by identity");
  const overwritten = mapVersions[1].insert(mapVersions[1].keys()[0], "replaced");
  assert.strictEqual(overwritten.size, mapVersions[1].size, "overwrite keeps the size");
  assert.strictEqual(overwritten.get(mapVersions[1].keys()[0]), "replaced");
  assert.notStrictEqual(mapVersions[1].get(mapVersions[1].keys()[0]), "replaced", "the old version keeps the old value");
  const untouched = map.remove(123456);
  assert.strictEqual(untouched, map, "removing a missing key returns the same version");

  // deletions with historical reads
  let shrinking = map;
  const shrinkOracles = [new Map(mapOracles[mapOracles.length - 1])];
  const removalOrder = [...new Map(mapOracles[mapOracles.length - 1]).keys()].sort((a, b) => a - b);
  for (const key of removalOrder) {
    shrinking = shrinking.remove(key);
    const snapshot = new Map(shrinkOracles[shrinkOracles.length - 1]);
    snapshot.delete(key);
    shrinking.validate();
    assert.strictEqual(shrinking.size, snapshot.size, `size after removing ${key}`);
    for (const [k, v] of snapshot) assert.strictEqual(shrinking.get(k), v, `value ${k} after removing ${key}`);
    assert.strictEqual(shrinking.get(key), undefined, `removed key ${key}`);
    shrinkOracles.push(snapshot);
  }
  assert.strictEqual(shrinking.size, 0);
  assert.strictEqual(shrinking.root, null);
  assert.strictEqual(map.size, removalOrder.length, "the pre-deletion version is untouched");
  assert.strictEqual(map.get(removalOrder[0]), 0, "and it still reads the first key's value");

  // path copying budget for the treap: O(updates * log n) allocations, not O(updates * n)
  resetAllocationCount();
  let budget = new PersistentMap();
  const keys = 512;
  let naiveNodes = 0;                                            // copying the whole map each time
  for (let i = 1; i <= keys; i++) {
    naiveNodes += i;
    budget = budget.insert(i, i);
  }
  const copyingNodes = allocationCount;
  assert.ok(copyingNodes < naiveNodes, `path copying (${copyingNodes}) must beat a full copy (${naiveNodes})`);
  assert.ok(copyingNodes <= keys * 8 * Math.ceil(Math.log2(keys + 1)), "allocations stay on the log path per update");
  budget.validate();
  console.log(`treap: ${keys} inserts, path copying allocated ${copyingNodes} nodes, naive full copy needs ${naiveNodes}`);

  // --- persistent vector: one copied node per level, identity sharing on the other branch ---
  const vector = new PersistentVector(64);
  vector.validate();
  assert.strictEqual(vector.slots, 64);
  assert.strictEqual(vector.levels, 6);
  assert.strictEqual(vector.liveSlots, 0);
  assert.throws(() => new PersistentVector(48), RangeError);
  assert.throws(() => new PersistentVector(0), RangeError);
  assert.throws(() => vector.get(64), RangeError);
  assert.throws(() => vector.get(-1), RangeError);
  assert.throws(() => vector.set(64, 0), RangeError);
  const vectorVersions = [vector];
  const vectorOracles = [new Map()];
  const next = xorshift32(0xfeed);
  for (let step = 0; step < 60; step++) {
    const index = Math.floor(next() * 64);
    const value = step * 3 + 1;
    const before = vectorVersions[vectorVersions.length - 1];
    const after = before.set(index, value);
    after.validate();
    assert.strictEqual(after.allocations, before.levels, `one copied node per level at step ${step}`);
    const snapshot = new Map(vectorOracles[vectorOracles.length - 1]);
    snapshot.set(index, value);
    vectorVersions.push(after);
    vectorOracles.push(snapshot);
  }
  for (const [index, version] of vectorVersions.entries()) {
    const oracle = vectorOracles[index];
    assert.deepStrictEqual(version.toArray(), [...oracle.entries()].sort((a, b) => a[0] - b[0]), `vector version ${index}`);
    assert.strictEqual(version.liveSlots, oracle.size, `live slots of version ${index}`);
  }
  const vectorOld = vectorVersions[5];
  const vectorOldArray = vectorOld.toArray();
  const oldZero = vectorOld.get(0);
  const vectorNew = vectorOld.set(0, "new");
  assert.deepStrictEqual(vectorOld.toArray(), vectorOldArray, "an old vector version is immutable");
  assert.strictEqual(vectorNew.get(0), "new");
  assert.strictEqual(vectorOld.get(0), oldZero, "the old version still reads its own slot 0");

  resetAllocationCount();
  let shared = new PersistentVector(1024);
  const baseline = allocationCount;
  shared = shared.set(3, "a").set(900, "b");
  const vectorNodes = allocationCount - baseline;
  assert.strictEqual(vectorNodes, 2 * Math.log2(1024), "two updates copy two paths");
  assert.ok(vectorNodes < shared.slots, "sharing keeps the allocation far below a full copy");
  console.log(`vector: 1024 slots (2047 nodes), two updates allocated ${vectorNodes} nodes`);

  console.log("PASS");
}
