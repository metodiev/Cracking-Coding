/** Task 433 - Implement order statistics tree using balanced BST (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The balanced BST is a treap: a BST ordered by key plus a min-heap ordered by priority, kept
 * balanced by rotations on insert and by merging the two children on delete. Each node caches the
 * size of its subtree and a multiplicity count, so it acts as a multiset: rank(x) returns the
 * number of stored elements strictly below x and select(k) returns the k-th smallest (1-based,
 * counting multiplicities). Priorities are drawn from the shared LCG stream, which makes the tree
 * shape - and therefore every assertion - fully deterministic. Assertions compare insert/delete/
 * rank/select/count against a sorted-list oracle on LCG operation scripts and re-check the
 * heap/BST/size invariants with a structural validator.
 */
"use strict";

class Node {
  constructor(key, priority) {
    this.key = key;
    this.priority = priority;
    this.count = 1;
    this.subtreeSize = 1;
    this.left = null;
    this.right = null;
  }
}

class OrderStatisticsTree {
  constructor() {
    this.root = null;
    this.total = 0;
  }

  size() {
    return this.total;
  }

  static _sizeOf(node) {
    return node === null ? 0 : node.subtreeSize;
  }

  static _update(node) {
    node.subtreeSize =
      OrderStatisticsTree._sizeOf(node.left) + node.count + OrderStatisticsTree._sizeOf(node.right);
  }

  _rotateRight(node) {
    const child = node.left;
    node.left = child.right;
    child.right = node;
    OrderStatisticsTree._update(node);
    OrderStatisticsTree._update(child);
    return child;
  }

  _rotateLeft(node) {
    const child = node.right;
    node.right = child.left;
    child.left = node;
    OrderStatisticsTree._update(node);
    OrderStatisticsTree._update(child);
    return child;
  }

  insert(key, priority) {
    this.root = this._insert(this.root, key, priority);
    this.total++;
  }

  _insert(node, key, priority) {
    if (node === null) return new Node(key, priority);
    if (key === node.key) {
      node.count++;
      node.subtreeSize++;
      return node;
    }
    if (key < node.key) {
      node.left = this._insert(node.left, key, priority);
      if (node.left.priority < node.priority) return this._rotateRight(node);
    } else {
      node.right = this._insert(node.right, key, priority);
      if (node.right.priority < node.priority) return this._rotateLeft(node);
    }
    OrderStatisticsTree._update(node);
    return node;
  }

  delete(key) {
    const result = this._delete(this.root, key);
    this.root = result.node;
    if (result.removed) this.total--;
    return result.removed;
  }

  _delete(node, key) {
    if (node === null) return { node: null, removed: false };
    if (key === node.key) {
      if (node.count > 1) {
        node.count--;
        node.subtreeSize--;
        return { node, removed: true };
      }
      return { node: this._merge(node.left, node.right), removed: true };
    }
    const result = key < node.key ? this._delete(node.left, key) : this._delete(node.right, key);
    if (key < node.key) node.left = result.node;
    else node.right = result.node;
    OrderStatisticsTree._update(node);
    return { node, removed: result.removed };
  }

  _merge(left, right) {
    if (left === null) return right;
    if (right === null) return left;
    if (left.priority <= right.priority) {
      left.right = this._merge(left.right, right);
      OrderStatisticsTree._update(left);
      return left;
    }
    right.left = this._merge(left, right.left);
    OrderStatisticsTree._update(right);
    return right;
  }

  rank(key) {
    return this._rank(this.root, key);
  }

  _rank(node, key) {
    if (node === null) return 0;
    if (key <= node.key) return this._rank(node.left, key);
    return OrderStatisticsTree._sizeOf(node.left) + node.count + this._rank(node.right, key);
  }

  select(k) {
    if (k < 1 || k > this.total) throw new RangeError("select index out of range");
    return this._select(this.root, k);
  }

  _select(node, k) {
    const leftSize = OrderStatisticsTree._sizeOf(node.left);
    if (k <= leftSize) return this._select(node.left, k);
    if (k <= leftSize + node.count) return node.key;
    return this._select(node.right, k - leftSize - node.count);
  }

  countOf(key) {
    let node = this.root;
    while (node !== null) {
      if (key === node.key) return node.count;
      node = key < node.key ? node.left : node.right;
    }
    return 0;
  }

  inorder() {
    const out = [];
    const walk = (node) => {
      if (node === null) return;
      walk(node.left);
      for (let i = 0; i < node.count; i++) out.push(node.key);
      walk(node.right);
    };
    walk(this.root);
    return out;
  }

  validate() {
    const walk = (node, low, high) => {
      if (node === null) return 0;
      if (!(low < node.key && node.key < high)) return -1;
      const leftSize = walk(node.left, low, node.key);
      const rightSize = walk(node.right, node.key, high);
      if (leftSize < 0 || rightSize < 0) return -1;
      if (node.subtreeSize !== leftSize + node.count + rightSize) return -1;
      if (node.left !== null && node.left.priority < node.priority) return -1;
      if (node.right !== null && node.right.priority < node.priority) return -1;
      return node.subtreeSize;
    };
    return walk(this.root, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY) === this.total;
  }
}

function sortedInsert(sortedValues, value) {
  let lo = 0;
  let hi = sortedValues.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedValues[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  sortedValues.splice(lo, 0, value);
}

function lowerBound(sortedValues, value) {
  let lo = 0;
  let hi = sortedValues.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedValues[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

module.exports = { Node, OrderStatisticsTree, sortedInsert, lowerBound };

if (require.main === module) {
  const assert = require("node:assert");
  let tree = new OrderStatisticsTree();
  assert.strictEqual(tree.size(), 0);
  for (const [key, priority] of [[5, 100], [1, 90], [9, 80], [3, 70], [3, 110]]) {
    tree.insert(key, priority);
  }
  assert.deepStrictEqual(tree.inorder(), [1, 3, 3, 5, 9]);
  assert.strictEqual(tree.size(), 5);
  assert.ok(tree.validate());
  assert.deepStrictEqual([1, 2, 3, 4, 5].map((k) => tree.select(k)), [1, 3, 3, 5, 9]);
  assert.strictEqual(tree.rank(1), 0);
  assert.strictEqual(tree.rank(3), 1);
  assert.strictEqual(tree.rank(4), 3);
  assert.strictEqual(tree.rank(9), 4);
  assert.strictEqual(tree.countOf(3), 2);
  assert.strictEqual(tree.countOf(4), 0);
  assert.strictEqual(tree.delete(3), true);
  assert.deepStrictEqual(tree.inorder(), [1, 3, 5, 9]);
  assert.strictEqual(tree.select(3), 5);
  assert.strictEqual(tree.delete(100), false);
  assert.ok(tree.validate());

  tree = new OrderStatisticsTree();
  let oracle = [];
  let seed = 4331;
  for (let step = 0; step < 400; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 5;
    if (op <= 1) {
      seed = (seed * 75 + 74) % 65537;
      const key = seed % 60;
      seed = (seed * 75 + 74) % 65537;
      tree.insert(key, seed);
      sortedInsert(oracle, key);
    } else if (op === 2) {
      seed = (seed * 75 + 74) % 65537;
      const key = seed % 60;
      const index = oracle.indexOf(key);
      const expected = index !== -1;
      if (expected) oracle.splice(index, 1);
      assert.strictEqual(tree.delete(key), expected, `delete step ${step}`);
    } else if (op === 3) {
      seed = (seed * 75 + 74) % 65537;
      const key = seed % 60;
      assert.strictEqual(tree.rank(key), lowerBound(oracle, key), `rank step ${step}`);
      assert.strictEqual(tree.countOf(key), oracle.filter((v) => v === key).length, `count step ${step}`);
    } else if (oracle.length > 0) {
      seed = (seed * 75 + 74) % 65537;
      const k = 1 + (seed % oracle.length);
      assert.strictEqual(tree.select(k), oracle[k - 1], `select step ${step}`);
    }
    assert.strictEqual(tree.size(), oracle.length, `size step ${step}`);
    if (step % 25 === 0) {
      assert.deepStrictEqual(tree.inorder(), oracle, `inorder step ${step}`);
      assert.ok(tree.validate(), `validate step ${step}`);
    }
  }
  assert.ok(oracle.length > 0);
  assert.deepStrictEqual(tree.inorder(), oracle);
  assert.ok(tree.validate());
  console.log("PASS");
}
