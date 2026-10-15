/** Task 236 - Count nodes at distance K from target in binary tree. (Tier: Intermediate, CTCI Ch 4) */
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
 * Distance counts edges, so k = 0 is the target. A BFS records every node's parent;
 * a second BFS radiates outward from the target k steps and counts the frontier.
 * O(n) time. A missing target, empty tree or k beyond the diameter return 0; for
 * repeated values the first node in level order is the target.
 */
function countNodesAtDistance(root, targetVal, k) {
  if (root === null || k < 0) return 0;
  const parent = new Map();
  parent.set(root, null);
  let target = null;
  const q = [root];
  while (q.length) {
    const node = q.shift();
    if (target === null && node.val === targetVal) target = node;
    for (const child of [node.left, node.right]) {
      if (child !== null) {
        parent.set(child, node);
        q.push(child);
      }
    }
  }
  if (target === null) return 0;
  const seen = new Set([target]);
  let frontier = [target];                        // nodes exactly `step` edges away
  for (let step = 0; step < k; step++) {
    const next = [];
    for (const node of frontier) {
      for (const nxt of [node.left, node.right, parent.get(node)]) {
        if (nxt !== null && !seen.has(nxt)) {
          seen.add(nxt);
          next.push(nxt);
        }
      }
    }
    frontier = next;
    if (frontier.length === 0) return 0;
  }
  return frontier.length;
}

/** Root-to-target path via BFS parent pointers (same target rule as the main one). */
function findPath(root, targetVal) {
  const parent = new Map([[root, null]]);
  const q = [root];
  let target = null;
  while (q.length && target === null) {
    const node = q.shift();
    if (node.val === targetVal) {
      target = node;
      break;
    }
    for (const child of [node.left, node.right]) {
      if (child !== null) {
        parent.set(child, node);
        q.push(child);
      }
    }
  }
  if (target === null) return null;
  const path = [];
  for (let n = target; n !== null; n = parent.get(n)) path.push(n);
  return path.reverse();
}

/** O(n * h) reference: distance = root paths without their common prefix. */
function bruteCount(root, targetVal, k) {
  if (root === null || k < 0) return 0;
  const targetPath = findPath(root, targetVal);
  if (targetPath === null) return 0;
  const dfs = (node, path) => {
    path.push(node);
    let common = 0;
    while (common < path.length && common < targetPath.length && path[common] === targetPath[common]) common++;
    let count = path.length - common + targetPath.length - common === k ? 1 : 0;
    if (node.left !== null) count += dfs(node.left, path);
    if (node.right !== null) count += dfs(node.right, path);
    path.pop();
    return count;
  };
  return dfs(root, []);
}

module.exports = { TreeNode, buildTree, countNodesAtDistance, bruteCount };

if (require.main === module) {
  const assert = require("node:assert");
  const classic = buildTree([3, 5, 1, 6, 2, 0, 8, null, null, 7, 4]);
  assert.strictEqual(countNodesAtDistance(classic, 5, 0), 1);   // target itself
  assert.strictEqual(countNodesAtDistance(classic, 5, 1), 3);   // {3, 6, 2}
  assert.strictEqual(countNodesAtDistance(classic, 5, 2), 3);   // {1, 7, 4}
  assert.strictEqual(countNodesAtDistance(classic, 5, 3), 2);   // {0, 8}
  assert.strictEqual(countNodesAtDistance(classic, 5, 4), 0);
  assert.strictEqual(countNodesAtDistance(classic, 3, 0), 1);
  assert.strictEqual(countNodesAtDistance(classic, 3, 2), 4);   // {6, 2, 0, 8}
  assert.strictEqual(countNodesAtDistance(classic, 3, 3), 2);   // {7, 4}
  assert.strictEqual(countNodesAtDistance(classic, 99, 1), 0);  // missing target

  const skewed = buildTree([1, 2, null, 3, null, 4]);
  assert.strictEqual(countNodesAtDistance(skewed, 1, 3), 1);    // root to deepest in chain
  assert.strictEqual(countNodesAtDistance(skewed, 1, 4), 0);
  assert.strictEqual(countNodesAtDistance(buildTree([7]), 7, 0), 1);
  assert.strictEqual(countNodesAtDistance(buildTree([7]), 7, 1), 0);
  assert.strictEqual(countNodesAtDistance(null, 1, 0), 0);
  assert.strictEqual(countNodesAtDistance(buildTree([1, 2, 1]), 1, 1), 2);
  assert.strictEqual(countNodesAtDistance(buildTree([1, 2, 1]), 1, 2), 0);

  // deterministic LCG so the cross-check is reproducible
  let seed = 236;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const prune = (node) => {
    if (node === null || rand() < 0.25) return null;
    node.left = prune(node.left);
    node.right = prune(node.right);
    return node;
  };
  for (let test = 0; test < 200; test++) {       // cross-check vs path-based brute force
    const n = Math.floor(rand() * 13);
    const nodes = [];
    for (let i = 0; i < n; i++) nodes.push(new TreeNode(i + 1));
    for (let i = 0; i < n; i++) {
      if (2 * i + 1 < n) nodes[i].left = nodes[2 * i + 1];
      if (2 * i + 2 < n) nodes[i].right = nodes[2 * i + 2];
    }
    const root = n === 0 ? null : prune(nodes[0]);
    const reachable = [];
    if (root !== null) {
      const stack = [root];
      while (stack.length) {
        const node = stack.pop();
        reachable.push(node);
        if (node.left !== null) stack.push(node.left);
        if (node.right !== null) stack.push(node.right);
      }
    }
    const targetVal = reachable.length === 0
      ? 1
      : reachable[Math.floor(rand() * reachable.length)].val;
    for (let k = 0; k <= n + 2; k++) {
      assert.strictEqual(countNodesAtDistance(root, targetVal, k), bruteCount(root, targetVal, k),
        `brute force: target=${targetVal} k=${k}`);
    }
  }
  console.log("PASS");
}
