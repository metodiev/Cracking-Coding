/** Task 434 - Implement scapegoat tree basics (rebuild heuristic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A scapegoat tree is an unbalanced BST that stays near-balanced by whole-subtree rebuilds instead
 * of rotations. Documented constants: balance factor alpha = 0.75 (a node may not have a child
 * subtree larger than alpha * its own size) and the delete trigger size < alpha * max_size, where
 * max_size is the peak size since the last full rebuild. Insert walks down, then walks back up
 * updating cached sizes and picks the deepest ancestor whose child exceeds alpha * its size - the
 * scapegoat - and rebuilds its subtree into a perfectly balanced half-split. Delete is physical:
 * the key is replaced by its in-order successor and sizes are repaired on the way up; falling below
 * alpha * max_size rebuilds the whole tree. Keys are unique (duplicate inserts are no-ops), matching
 * the sorted-list oracle used on LCG operation scripts. Rebuilds are counted: the adversarial
 * ascending insert 1..200 triggers at least one rebuild (a final assert documents the count), and
 * every state is checked for BST order, cached sizes and O(log n) height.
 */
"use strict";

class Node {
  constructor(key) {
    this.key = key;
    this.left = null;
    this.right = null;
    this.size = 1;
  }
}

class ScapegoatTree {
  constructor(alpha = 0.75) {
    this.alpha = alpha;
    this.root = null;
    this.total = 0;
    this.maxSize = 0;
    this.rebuilds = 0;
  }

  size() {
    return this.total;
  }

  contains(key) {
    let node = this.root;
    while (node !== null) {
      if (key === node.key) return true;
      node = key < node.key ? node.left : node.right;
    }
    return false;
  }

  insert(key) {
    if (this.root === null) {
      this.root = new Node(key);
      this.total = 1;
      this.maxSize = 1;
      return true;
    }
    const path = [];
    let node = this.root;
    for (;;) {
      path.push(node);
      if (key === node.key) return false;
      if (key < node.key) {
        if (node.left === null) {
          node.left = new Node(key);
          break;
        }
        node = node.left;
      } else {
        if (node.right === null) {
          node.right = new Node(key);
          break;
        }
        node = node.right;
      }
    }
    let scapegoatIndex = -1;
    for (let i = path.length - 1; i >= 0; i--) {
      const current = path[i];
      current.size++;
      if (scapegoatIndex === -1) {
        const left = current.left === null ? 0 : current.left.size;
        const right = current.right === null ? 0 : current.right.size;
        if (left > this.alpha * current.size || right > this.alpha * current.size) {
          scapegoatIndex = i;
        }
      }
    }
    this.total++;
    this.maxSize = Math.max(this.maxSize, this.total);
    if (scapegoatIndex >= 0) this._rebuildAt(path, scapegoatIndex);
    return true;
  }

  delete(key) {
    if (!this.contains(key)) return false;
    this.root = this._delete(this.root, key);
    this.total--;
    if (this.total < this.alpha * this.maxSize) {
      const keys = this._collect(this.root);
      this.root = this._build(keys, 0, keys.length - 1);
      this.maxSize = this.total;
      this.rebuilds++;
    }
    return true;
  }

  _delete(node, key) {
    if (key < node.key) {
      node.left = this._delete(node.left, key);
    } else if (key > node.key) {
      node.right = this._delete(node.right, key);
    } else {
      return this._deleteNode(node);
    }
    this._update(node);
    return node;
  }

  _deleteNode(node) {
    if (node.left === null) return node.right;
    if (node.right === null) return node.left;
    let successor = node.right;
    while (successor.left !== null) successor = successor.left;
    node.key = successor.key;
    node.right = this._delete(node.right, successor.key);
    this._update(node);
    return node;
  }

  inorder() {
    const out = [];
    const stack = [];
    let node = this.root;
    while (stack.length > 0 || node !== null) {
      while (node !== null) {
        stack.push(node);
        node = node.left;
      }
      node = stack.pop();
      out.push(node.key);
      node = node.right;
    }
    return out;
  }

  height() {
    const depth = (node) => {
      if (node === null) return 0;
      return 1 + Math.max(depth(node.left), depth(node.right));
    };
    return depth(this.root);
  }

  validate() {
    const walk = (node, low, high) => {
      if (node === null) return 0;
      if (!(low < node.key && node.key < high)) return -1;
      const leftSize = walk(node.left, low, node.key);
      const rightSize = walk(node.right, node.key, high);
      if (leftSize < 0 || rightSize < 0) return -1;
      if (node.size !== 1 + leftSize + rightSize) return -1;
      return node.size;
    };
    return walk(this.root, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY) === this.total;
  }

  _collect(node) {
    const out = [];
    const stack = [];
    while (stack.length > 0 || node !== null) {
      while (node !== null) {
        stack.push(node);
        node = node.left;
      }
      node = stack.pop();
      out.push(node.key);
      node = node.right;
    }
    return out;
  }

  _update(node) {
    node.size = 1 + (node.left === null ? 0 : node.left.size) + (node.right === null ? 0 : node.right.size);
  }

  _rebuildAt(path, index) {
    const node = path[index];
    const rebuilt = this._build(this._collect(node), 0, node.size - 1);
    if (index === 0) {
      this.root = rebuilt;
    } else {
      const parent = path[index - 1];
      if (parent.left === node) parent.left = rebuilt;
      else parent.right = rebuilt;
    }
    this.rebuilds++;
  }

  _build(keys, lo, hi) {
    if (lo > hi) return null;
    const mid = (lo + hi) >> 1;
    const node = new Node(keys[mid]);
    node.left = this._build(keys, lo, mid - 1);
    node.right = this._build(keys, mid + 1, hi);
    this._update(node);
    return node;
  }
}

function heightBound(size) {
  return 3 * Math.ceil(Math.log2(size + 1)) + 3;
}

module.exports = { Node, ScapegoatTree, heightBound };

if (require.main === module) {
  const assert = require("node:assert");
  let tree = new ScapegoatTree();
  assert.strictEqual(tree.size(), 0);
  assert.deepStrictEqual(tree.inorder(), []);
  assert.ok(tree.validate());
  assert.strictEqual(tree.delete(0), false);

  for (const key of [10, 5, 15, 3]) assert.strictEqual(tree.insert(key), true);
  assert.strictEqual(tree.insert(10), false);
  assert.deepStrictEqual(tree.inorder(), [3, 5, 10, 15]);
  assert.ok(tree.contains(15) && !tree.contains(4));
  assert.ok(tree.validate());
  assert.strictEqual(tree.delete(5), true);
  assert.strictEqual(tree.delete(5), false);
  assert.deepStrictEqual(tree.inorder(), [3, 10, 15]);
  assert.ok(tree.validate());

  tree = new ScapegoatTree();
  for (let key = 1; key <= 200; key++) tree.insert(key);
  assert.deepStrictEqual(tree.inorder(), Array.from({ length: 200 }, (_, i) => i + 1));
  assert.ok(tree.rebuilds >= 1, `ascending rebuilds ${tree.rebuilds}`);
  assert.ok(tree.height() <= heightBound(tree.size()), `ascending height ${tree.height()}`);
  for (let key = 2; key <= 200; key += 2) assert.strictEqual(tree.delete(key), true);
  assert.deepStrictEqual(tree.inorder(), Array.from({ length: 100 }, (_, i) => 2 * i + 1));
  assert.ok(tree.validate());
  for (let key = 1; key <= 200; key += 2) assert.strictEqual(tree.delete(key), true);
  assert.strictEqual(tree.size(), 0);
  assert.deepStrictEqual(tree.inorder(), []);
  assert.ok(tree.validate());

  tree = new ScapegoatTree();
  let oracle = [];
  let seed = 4340;
  for (let step = 0; step < 400; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 3;
    seed = (seed * 75 + 74) % 65537;
    const key = seed % 100;
    if (op === 0) {
      const got = tree.insert(key);
      const expected = !oracle.includes(key);
      if (expected) {
        oracle.push(key);
        oracle.sort((a, b) => a - b);
      }
      assert.strictEqual(got, expected, `insert step ${step}`);
    } else {
      const got = tree.delete(key);
      const index = oracle.indexOf(key);
      const expected = index !== -1;
      if (expected) oracle.splice(index, 1);
      assert.strictEqual(got, expected, `delete step ${step}`);
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
  assert.ok(tree.height() <= heightBound(tree.size()), `final height ${tree.height()}`);
  console.log("PASS");
}
