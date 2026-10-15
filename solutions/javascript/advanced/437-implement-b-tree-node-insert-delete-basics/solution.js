/** Task 437 - Implement B-tree node insert/delete basics. (Tier: Advanced, CTCI Ch 14)
 *
 * A B-tree of order m = 4 (minimum degree t = 2): every node holds at most 2t-1 = 3 keys and, when
 * internal, 2..4 children; non-root nodes keep at least t-1 = 1 key, and all leaves sit at the same
 * depth. Multiset semantics are obtained by storing collision-free composite keys
 * key = (value << 20) | serial: each insertion gets a fresh serial, so the textbook unique-key
 * insert/delete apply unchanged while the sorted traversal still yields a sorted multiset of values.
 * Insert uses the preemptive split rule - a full node (3 keys) is split around its median before
 * descending, the median rises into the parent and the upper half becomes a new right sibling.
 * Delete follows CLRS: a key found in an internal node is replaced by its in-order predecessor or
 * successor; before descending into a child that holds only t-1 keys the child is repaired by
 * borrowing a key from a sibling (rotation through the parent) or by merging with a sibling - the
 * documented merge/borrow counter is asserted to be positive after the LCG workload. The sorted
 * traversal is compared against a sorted multiset oracle on LCG operation scripts, and structure
 * invariants (key order, occupancy, uniform leaf depth) are re-validated throughout.
 */
"use strict";

const T = 2;
const SHIFT = 20;

class BTreeNode {
  constructor() {
    this.keys = [];
    this.children = [];
  }
}

function lowerBound(keys, key) {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (keys[mid] < key) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

class BTree {
  constructor() {
    this.root = new BTreeNode();
    this.size = 0;
    this.nextSerial = 0;
    this.splits = 0;
    this.merges = 0;
    this.borrows = 0;
  }

  insert(value) {
    const key = (value << SHIFT) | this.nextSerial;
    this.nextSerial++;
    if (this.root.keys.length === 2 * T - 1) {
      const newRoot = new BTreeNode();
      newRoot.children.push(this.root);
      this._splitChild(newRoot, 0);
      this.root = newRoot;
    }
    this._insertNonfull(this.root, key);
    this.size++;
  }

  _splitChild(parent, index) {
    const node = parent.children[index];
    const right = new BTreeNode();
    const median = node.keys[T - 1];
    right.keys = node.keys.slice(T);
    node.keys = node.keys.slice(0, T - 1);
    if (node.children.length > 0) {
      right.children = node.children.slice(T);
      node.children = node.children.slice(0, T);
    }
    parent.keys.splice(index, 0, median);
    parent.children.splice(index + 1, 0, right);
    this.splits++;
  }

  _insertNonfull(node, key) {
    const index = lowerBound(node.keys, key);
    if (node.children.length > 0) {
      if (node.children[index].keys.length === 2 * T - 1) {
        this._splitChild(node, index);
        if (key > node.keys[index]) {
          this._insertNonfull(node.children[index + 1], key);
          return;
        }
      }
      this._insertNonfull(node.children[index], key);
    } else {
      node.keys.splice(index, 0, key);
    }
  }

  _lowerBoundKey(node, base) {
    const index = lowerBound(node.keys, base);
    if (node.children.length > 0) {
      const found = this._lowerBoundKey(node.children[index], base);
      if (found !== null) return found;
      if (index < node.keys.length) return node.keys[index];
      return null;
    }
    return index < node.keys.length ? node.keys[index] : null;
  }

  contains(value) {
    const key = this._lowerBoundKey(this.root, value << SHIFT);
    return key !== null && (key >> SHIFT) === value;
  }

  delete(value) {
    const key = this._lowerBoundKey(this.root, value << SHIFT);
    if (key === null || (key >> SHIFT) !== value) return false;
    this._delete(this.root, key);
    if (this.root.keys.length === 0 && this.root.children.length > 0) {
      this.root = this.root.children[0];
    }
    this.size--;
    return true;
  }

  _delete(node, key) {
    const index = lowerBound(node.keys, key);
    if (index < node.keys.length && node.keys[index] === key) {
      if (node.children.length === 0) {
        node.keys.splice(index, 1);
      } else {
        this._deleteInternal(node, index);
      }
      return;
    }
    if (node.children.length === 0) return;
    const flag = index === node.keys.length;
    let childIndex = index;
    if (node.children[childIndex].keys.length === T - 1) {
      this._fill(node, childIndex);
      if (flag && childIndex > node.keys.length) childIndex--;
    }
    this._delete(node.children[childIndex], key);
  }

  _deleteInternal(node, index) {
    const key = node.keys[index];
    const left = node.children[index];
    const right = node.children[index + 1];
    if (left.keys.length > T - 1) {
      const predecessor = this._maxKey(left);
      node.keys[index] = predecessor;
      this._delete(left, predecessor);
    } else if (right.keys.length > T - 1) {
      const successor = this._minKey(right);
      node.keys[index] = successor;
      this._delete(right, successor);
    } else {
      this._merge(node, index);
      this._delete(node.children[index], key);
    }
  }

  _fill(node, index) {
    if (index > 0 && node.children[index - 1].keys.length > T - 1) {
      this._borrowFromLeft(node, index);
    } else if (index < node.keys.length && node.children[index + 1].keys.length > T - 1) {
      this._borrowFromRight(node, index);
    } else if (index < node.keys.length) {
      this._merge(node, index);
    } else {
      this._merge(node, index - 1);
    }
  }

  _borrowFromLeft(node, index) {
    const child = node.children[index];
    const left = node.children[index - 1];
    child.keys.unshift(node.keys[index - 1]);
    node.keys[index - 1] = left.keys.pop();
    if (left.children.length > 0) child.children.unshift(left.children.pop());
    this.borrows++;
  }

  _borrowFromRight(node, index) {
    const child = node.children[index];
    const right = node.children[index + 1];
    child.keys.push(node.keys[index]);
    node.keys[index] = right.keys.shift();
    if (right.children.length > 0) child.children.push(right.children.shift());
    this.borrows++;
  }

  _merge(node, index) {
    const left = node.children[index];
    const right = node.children[index + 1];
    left.keys.push(node.keys.splice(index, 1)[0]);
    left.keys = left.keys.concat(right.keys);
    if (left.children.length > 0) left.children = left.children.concat(right.children);
    node.children.splice(index + 1, 1);
    this.merges++;
  }

  _minKey(node) {
    while (node.children.length > 0) node = node.children[0];
    return node.keys[0];
  }

  _maxKey(node) {
    while (node.children.length > 0) node = node.children[node.children.length - 1];
    return node.keys[node.keys.length - 1];
  }

  inorderValues() {
    const out = [];
    const walk = (node) => {
      for (let i = 0; i < node.keys.length; i++) {
        if (node.children.length > 0) walk(node.children[i]);
        out.push(node.keys[i] >> SHIFT);
      }
      if (node.children.length > 0) walk(node.children[node.keys.length]);
    };
    walk(this.root);
    return out;
  }

  validate() {
    const walk = (node, isRoot, depth, leafDepth) => {
      if (node.keys.length > 2 * T - 1) return -1;
      for (let i = 1; i < node.keys.length; i++) {
        if (node.keys[i - 1] >= node.keys[i]) return -1;
      }
      if (node.children.length > 0) {
        if (node.children.length !== node.keys.length + 1) return -1;
        if (node.children.length < (isRoot ? 2 : T)) return -1;
        for (const child of node.children) {
          leafDepth = walk(child, false, depth + 1, leafDepth);
          if (leafDepth < 0) return -1;
        }
        return leafDepth;
      }
      if (!isRoot && node.keys.length < T - 1) return -1;
      if (leafDepth !== -1 && leafDepth !== depth) return -1;
      return depth;
    };
    return walk(this.root, true, 0, -1) >= 0;
  }
}

function sortedInsert(oracle, value) {
  oracle.splice(lowerBound(oracle, value), 0, value);
}

module.exports = { T, SHIFT, BTreeNode, BTree, lowerBound, sortedInsert };

if (require.main === module) {
  const assert = require("node:assert");
  let tree = new BTree();
  assert.ok(tree.validate());
  assert.deepStrictEqual(tree.inorderValues(), []);
  assert.strictEqual(tree.delete(5), false);

  for (const value of [10, 20, 30, 40]) tree.insert(value);
  assert.ok(tree.splits >= 1, `root split ${tree.splits}`);
  assert.deepStrictEqual(tree.inorderValues(), [10, 20, 30, 40]);
  assert.ok(tree.validate());
  assert.ok(tree.contains(30) && !tree.contains(35));
  assert.strictEqual(tree.size, 4);

  tree.insert(10);
  tree.insert(10);
  assert.deepStrictEqual(tree.inorderValues(), [10, 10, 10, 20, 30, 40]);
  assert.strictEqual(tree.delete(10), true);
  assert.deepStrictEqual(tree.inorderValues(), [10, 10, 20, 30, 40]);
  assert.strictEqual(tree.delete(10), true);
  assert.strictEqual(tree.delete(10), true);
  assert.strictEqual(tree.contains(10), false);
  assert.strictEqual(tree.delete(10), false);
  assert.deepStrictEqual(tree.inorderValues(), [20, 30, 40]);

  for (const value of [35, 25, 15, 5, 45, 50, 55, 60]) tree.insert(value);
  assert.deepStrictEqual(tree.inorderValues(), [5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]);
  assert.ok(tree.validate());
  for (const value of [30, 5, 60, 20, 45, 15, 55, 25, 35, 50, 40]) {
    assert.strictEqual(tree.delete(value), true);
  }
  assert.deepStrictEqual(tree.inorderValues(), []);
  assert.strictEqual(tree.size, 0);
  assert.ok(tree.validate());

  tree = new BTree();
  let oracle = [];
  let seed = 4370;
  for (let step = 0; step < 300; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 4;
    seed = (seed * 75 + 74) % 65537;
    if (op <= 1) {
      const value = seed % 30;
      tree.insert(value);
      sortedInsert(oracle, value);
    } else if (op === 2 && oracle.length > 0) {
      const value = oracle.splice(seed % oracle.length, 1)[0];
      assert.strictEqual(tree.delete(value), true, `random delete ${step}`);
      const stillPresent = oracle.includes(value);
      assert.strictEqual(tree.delete(value), stillPresent, `random second delete ${step}`);
      if (stillPresent) oracle.splice(oracle.indexOf(value), 1);
    } else {
      const value = 100 + (seed % 5);
      assert.strictEqual(tree.delete(value), false, `random absent ${step}`);
    }
    assert.strictEqual(tree.size, oracle.length, `random size ${step}`);
    if (step % 20 === 0 || step === 299) {
      assert.deepStrictEqual(tree.inorderValues(), oracle, `random order ${step}`);
      assert.ok(tree.validate(), `random validate ${step}`);
    }
  }
  assert.ok(oracle.length > 0);
  assert.deepStrictEqual(tree.inorderValues(), oracle);
  assert.ok(tree.validate());
  assert.ok(tree.splits > 0 && tree.merges > 0, `counters ${tree.splits}/${tree.merges}/${tree.borrows}`);
  console.log("PASS");
}
