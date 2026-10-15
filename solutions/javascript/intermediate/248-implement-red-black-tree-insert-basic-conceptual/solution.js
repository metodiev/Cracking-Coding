/** Task 248 - Implement red-black tree insert (basic). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

const RED = true;
const BLACK = false;

class RBNode {
  constructor(key, red = RED) {
    this.key = key;
    this.red = red;
    this.left = null;
    this.right = null;
    this.parent = null;
  }
}

class RedBlackTree {
  constructor() {
    this.root = null;
  }

  rotateLeft(x) {
    const y = x.right;
    x.right = y.left;
    if (y.left) y.left.parent = x;
    y.parent = x.parent;
    if (!x.parent) this.root = y;
    else if (x === x.parent.left) x.parent.left = y;
    else x.parent.right = y;
    y.left = x;
    x.parent = y;
  }

  rotateRight(x) {
    const y = x.left;
    x.left = y.right;
    if (y.right) y.right.parent = x;
    y.parent = x.parent;
    if (!x.parent) this.root = y;
    else if (x === x.parent.right) x.parent.right = y;
    else x.parent.left = y;
    y.right = x;
    x.parent = y;
  }

  insert(key) {
    let parent = null;
    let node = this.root;
    while (node) {
      parent = node;
      if (key < node.key) node = node.left;
      else if (key > node.key) node = node.right;
      else return false;
    }
    const z = new RBNode(key);
    z.parent = parent;
    if (!parent) this.root = z;
    else if (key < parent.key) parent.left = z;
    else parent.right = z;
    this.fixup(z);
    return true;
  }

  fixup(z) {
    while (z.parent && z.parent.red) {
      let parent = z.parent;
      let grand = parent.parent;
      if (parent === grand.left) {
        const uncle = grand.right;
        if (uncle && uncle.red) {
          parent.red = BLACK;
          uncle.red = BLACK;
          grand.red = RED;
          z = grand;
        } else {
          if (z === parent.right) {
            z = parent;
            this.rotateLeft(z);
            parent = z.parent;
            grand = parent.parent;
          }
          parent.red = BLACK;
          grand.red = RED;
          this.rotateRight(grand);
        }
      } else {
        const uncle = grand.left;
        if (uncle && uncle.red) {
          parent.red = BLACK;
          uncle.red = BLACK;
          grand.red = RED;
          z = grand;
        } else {
          if (z === parent.left) {
            z = parent;
            this.rotateRight(z);
            parent = z.parent;
            grand = parent.parent;
          }
          parent.red = BLACK;
          grand.red = RED;
          this.rotateLeft(grand);
        }
      }
    }
    this.root.red = BLACK;
  }
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.key); inorder(node.right, out); }
  return out;
}

function noRedRed(node) {
  if (!node) return true;
  if (node.red) {
    if (node.left && node.left.red) return false;
    if (node.right && node.right.red) return false;
  }
  return noRedRed(node.left) && noRedRed(node.right);
}

function blackHeights(node, measure) {
  if (!node) return new Set([measure]);
  const sub = new Set([...blackHeights(node.left, measure), ...blackHeights(node.right, measure)]);
  const extra = node.red ? 0 : 1;
  return new Set([...sub].map((v) => v + extra));
}

module.exports = { RED, BLACK, RBNode, RedBlackTree, inorder, noRedRed, blackHeights };

if (require.main === module) {
  const assert = require("node:assert");
  let tree = new RedBlackTree();
  for (const key of [10, 20, 30]) assert.ok(tree.insert(key));
  assert.strictEqual(tree.root.key, 20);
  assert.ok(!tree.root.red);
  assert.ok(tree.root.left.red && tree.root.right.red);

  const orders = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [7, 3, 9, 1, 5, 8, 10, 2, 4, 6],
    [50, 20, 60, 10, 30, 55, 70, 5, 15, 25, 35],
  ];
  for (const order of orders) {
    tree = new RedBlackTree();
    for (const key of order) assert.ok(tree.insert(key));
    assert.deepStrictEqual(inorder(tree.root), [...order].sort((a, b) => a - b));
    assert.ok(!tree.root.red, "root black");
    assert.ok(noRedRed(tree.root), "no red-red");
    assert.strictEqual(blackHeights(tree.root, 0).size, 1, "uniform black height");
  }
  assert.strictEqual(tree.insert(50), false);
  console.log("PASS");
}
