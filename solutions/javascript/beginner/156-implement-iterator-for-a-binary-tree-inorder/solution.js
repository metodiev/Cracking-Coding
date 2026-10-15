/** Task 156 - Implement iterator for a binary tree (inorder). (Tier: Beginner, CTCI Ch 4) */
"use strict";

class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

/** Build a binary tree from level-order values (null marks a missing child). */
function buildTree(values) {
  if (values.length === 0 || values[0] === null) return null;
  const root = new TreeNode(values[0]);
  const q = [root];
  let i = 1;
  while (q.length && i < values.length) {
    const node = q.shift();
    if (i < values.length && values[i] !== null) {
      node.left = new TreeNode(values[i]);
      q.push(node.left);
    }
    i++;
    if (i < values.length && values[i] !== null) {
      node.right = new TreeNode(values[i]);
      q.push(node.right);
    }
    i++;
  }
  return root;
}

/**
 * In-order iterator with an explicit stack: push the left spine of the current
 * subtree; next() pops one node, then descends that node's right subtree's left
 * spine. O(1) amortised per step, O(h) extra memory.
 */
class InOrderIterator {
  constructor(root) {
    this.stack = [];
    this.pushLeftSpine(root);
  }

  pushLeftSpine(node) {
    while (node) {
      this.stack.push(node);
      node = node.left;
    }
  }

  hasNext() {
    return this.stack.length > 0;
  }

  /** Next in-order value; throws once exhausted. */
  next() {
    if (this.stack.length === 0) throw new Error("iterator exhausted");
    const node = this.stack.pop();
    this.pushLeftSpine(node.right);
    return node.val;
  }
}

module.exports = { TreeNode, buildTree, InOrderIterator };

if (require.main === module) {
  const assert = require("node:assert");
  // valid BST: in-order must be ascending
  const bst = buildTree([8, 3, 10, 1, 6, null, 14, null, null, 4, 7, 13]);
  const it = new InOrderIterator(bst);
  const values = [];
  while (it.hasNext()) values.push(it.next());
  assert.deepStrictEqual(values, [1, 3, 4, 6, 7, 8, 10, 13, 14]);
  for (let i = 1; i < values.length; i++) assert.ok(values[i - 1] < values[i], "ascending");

  // exhaustion: hasNext() is false and next() throws
  assert.strictEqual(it.hasNext(), false);
  assert.throws(() => it.next(), /iterator exhausted/);

  // empty tree is immediately exhausted
  const empty = new InOrderIterator(null);
  assert.strictEqual(empty.hasNext(), false);
  assert.throws(() => empty.next(), /iterator exhausted/);

  // hasNext is idempotent (peeks without consuming) and works one step at a time
  const it2 = new InOrderIterator(buildTree([1, 2, 3]));
  assert.ok(it2.hasNext() && it2.hasNext());
  assert.strictEqual(it2.next(), 2);
  assert.strictEqual(it2.next(), 1);
  assert.ok(it2.hasNext());
  assert.strictEqual(it2.next(), 3);
  assert.strictEqual(it2.hasNext(), false);

  // single node
  const it3 = new InOrderIterator(buildTree([5]));
  assert.ok(it3.hasNext());
  assert.strictEqual(it3.next(), 5);
  assert.strictEqual(it3.hasNext(), false);
  console.log("PASS");
}
