/** Task 239 - Connect next pointers in each node (perfect binary tree). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class Node {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
    this.next = null;
  }
}

/** Build a binary tree from level-order values (null marks a missing child). */
function buildTree(values) {
  if (values.length === 0 || values[0] === null) return null;
  const root = new Node(values[0]);
  const q = [root];
  let i = 1;
  while (q.length && i < values.length) {
    const node = q.shift();
    if (i < values.length && values[i] !== null) {
      node.left = new Node(values[i]);
      q.push(node.left);
    }
    i++;
    if (i < values.length && values[i] !== null) {
      node.right = new Node(values[i]);
      q.push(node.right);
    }
    i++;
  }
  return root;
}

/**
 * Every node gets next -> its right neighbour on the same level (null at level
 * ends). O(1) extra space: descend level by level through the already-set next
 * pointers of the parent level; for a parent p, p.left.next = p.right and, when
 * p.next exists, p.right.next = p.next.left. Returns the root.
 */
function connect(root) {
  if (root === null) return null;
  for (let leftmost = root; leftmost.left !== null; leftmost = leftmost.left) {
    for (let node = leftmost; node !== null; node = node.next) {
      node.left.next = node.right;
      if (node.next !== null) node.right.next = node.next.left;
    }
  }
  return root;
}

/** Read the levels through the next pointers (fails fast if a chain loops). */
function levelsViaNext(root) {
  const levels = [];
  for (let leftmost = root; leftmost !== null; leftmost = leftmost.left) {
    const level = [];
    for (let node = leftmost; node !== null; node = node.next) {
      level.push(node.val);
      if (level.length > 1000) throw new Error("next chain longer than expected");
    }
    levels.push(level);
  }
  return levels;
}

module.exports = { Node, buildTree, connect, levelsViaNext };

if (require.main === module) {
  const assert = require("node:assert");
  const root = connect(buildTree([1, 2, 3, 4, 5, 6, 7]));
  assert.deepStrictEqual(levelsViaNext(root), [[1], [2, 3], [4, 5, 6, 7]]);
  assert.strictEqual(root.next, null);
  assert.strictEqual(root.left.next, root.right);
  assert.strictEqual(root.right.next, null);
  assert.strictEqual(root.left.left.next, root.left.right);
  assert.strictEqual(root.left.right.next, root.right.left);
  assert.strictEqual(root.right.left.next, root.right.right);
  assert.strictEqual(root.right.right.next, null);

  const single = connect(buildTree([1]));
  assert.deepStrictEqual(levelsViaNext(single), [[1]]);
  assert.strictEqual(single.next, null);

  assert.strictEqual(connect(null), null);

  const big = connect(buildTree([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]));
  assert.deepStrictEqual(levelsViaNext(big),
    [[1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]);

  const again = connect(root);                    // connecting twice changes nothing
  assert.strictEqual(again, root);
  assert.deepStrictEqual(levelsViaNext(again), [[1], [2, 3], [4, 5, 6, 7]]);
  console.log("PASS");
}
