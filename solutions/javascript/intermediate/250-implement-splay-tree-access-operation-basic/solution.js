/** Task 250 - Implement splay tree access operation (basic). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class SplayNode {
  constructor(key) {
    this.key = key;
    this.left = null;
    this.right = null;
    this.parent = null;
  }
}

class SplayTree {
  constructor() {
    this.root = null;
  }

  replaceChild(parent, old, fresh) {
    if (!parent) this.root = fresh;
    else if (parent.left === old) parent.left = fresh;
    else parent.right = fresh;
    if (fresh) fresh.parent = parent;
  }

  rotateLeft(x) {
    const y = x.right;
    x.right = y.left;
    if (y.left) y.left.parent = x;
    const parent = x.parent;
    this.replaceChild(parent, x, y);
    y.left = x;
    x.parent = y;
  }

  rotateRight(x) {
    const y = x.left;
    x.left = y.right;
    if (y.right) y.right.parent = x;
    const parent = x.parent;
    this.replaceChild(parent, x, y);
    y.right = x;
    x.parent = y;
  }

  splay(x) {
    while (x.parent) {
      const parent = x.parent;
      const grand = parent.parent;
      if (!grand) {
        if (x === parent.left) this.rotateRight(parent);
        else this.rotateLeft(parent);
      } else if (x === parent.left && parent === grand.left) {
        this.rotateRight(grand);
        this.rotateRight(parent);
      } else if (x === parent.right && parent === grand.right) {
        this.rotateLeft(grand);
        this.rotateLeft(parent);
      } else if (x === parent.right && parent === grand.left) {
        this.rotateLeft(parent);
        this.rotateRight(grand);
      } else {
        this.rotateRight(parent);
        this.rotateLeft(grand);
      }
    }
    this.root = x;
  }

  insert(key) {
    let parent = null;
    let node = this.root;
    while (node) {
      parent = node;
      if (key < node.key) node = node.left;
      else if (key > node.key) node = node.right;
      else { this.splay(node); return false; }
    }
    const fresh = new SplayNode(key);
    fresh.parent = parent;
    if (!parent) this.root = fresh;
    else if (key < parent.key) parent.left = fresh;
    else parent.right = fresh;
    this.splay(fresh);
    return true;
  }

  access(key) {
    let node = this.root;
    while (node) {
      if (key === node.key) { this.splay(node); return true; }
      node = key < node.key ? node.left : node.right;
    }
    return false;
  }

  inorder() {
    const out = [];
    (function walk(node) {
      if (node) { walk(node.left); out.push(node.key); walk(node.right); }
    })(this.root);
    return out;
  }

  isBst() {
    const keys = this.inorder();
    for (let i = 1; i < keys.length; i++) if (keys[i - 1] >= keys[i]) return false;
    function check(node) {
      if (!node) return true;
      if (node.left && node.left.parent !== node) return false;
      if (node.right && node.right.parent !== node) return false;
      return check(node.left) && check(node.right);
    }
    return check(this.root);
  }
}

module.exports = { SplayNode, SplayTree };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = new SplayTree();
  for (const key of [50, 30, 70, 20, 40, 60, 80]) assert.ok(tree.insert(key));
  assert.strictEqual(tree.root.key, 80);
  assert.deepStrictEqual(tree.inorder(), [20, 30, 40, 50, 60, 70, 80]);
  assert.ok(tree.isBst());

  assert.ok(tree.access(20));
  assert.strictEqual(tree.root.key, 20);
  assert.ok(tree.isBst());

  assert.ok(tree.access(80));
  assert.strictEqual(tree.root.key, 80);
  assert.ok(tree.isBst());

  assert.ok(tree.access(50));
  assert.strictEqual(tree.root.key, 50);
  assert.deepStrictEqual(tree.inorder(), [20, 30, 40, 50, 60, 70, 80]);
  assert.ok(tree.isBst());

  assert.strictEqual(tree.access(999), false);
  assert.strictEqual(tree.root.key, 50);
  assert.strictEqual(tree.insert(50), false);
  assert.deepStrictEqual(tree.inorder(), [20, 30, 40, 50, 60, 70, 80]);
  assert.ok(tree.isBst());
  console.log("PASS");
}
