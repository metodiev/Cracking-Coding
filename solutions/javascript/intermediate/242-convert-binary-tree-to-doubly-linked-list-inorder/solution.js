/** Task 242 - Convert binary tree to doubly linked list (inorder). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

function buildTree(values) {
  if (values.length === 0 || values[0] === null) return null;
  const root = new TreeNode(values[0]);
  const q = [root];
  let i = 1;
  while (q.length && i < values.length) {
    const node = q.shift();
    if (i < values.length && values[i] !== null) { node.left = new TreeNode(values[i]); q.push(node.left); }
    i++;
    if (i < values.length && values[i] !== null) { node.right = new TreeNode(values[i]); q.push(node.right); }
    i++;
  }
  return root;
}

function preorder(node, out = []) {
  if (node) { out.push(node.val); preorder(node.left, out); preorder(node.right, out); }
  return out;
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.val); inorder(node.right, out); }
  return out;
}

function toDoublyLinkedList(root) {
  let last = null, head = null;

  function visit(node) {
    if (!node) return;
    visit(node.left);
    node.left = last;
    if (last) last.right = node;
    else head = node;
    last = node;
    visit(node.right);
  }

  visit(root);
  return head;
}

function toValues(head) {
  const out = [];
  for (let node = head; node; node = node.right) out.push(node.val);
  return out;
}

function toValuesReverse(head) {
  if (!head) return [];
  let node = head;
  while (node.right) node = node.right;
  const out = [];
  for (; node; node = node.left) out.push(node.val);
  return out;
}

module.exports = { TreeNode, buildTree, toDoublyLinkedList, toValues, toValuesReverse };

if (require.main === module) {
  const assert = require("node:assert");
  const head = toDoublyLinkedList(buildTree([4, 2, 5, 1, 3]));
  assert.deepStrictEqual(toValues(head), [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(toValuesReverse(head), [5, 4, 3, 2, 1]);
  assert.strictEqual(toDoublyLinkedList(null), null);
  const single = toDoublyLinkedList(buildTree([7]));
  assert.deepStrictEqual(toValues(single), [7]);
  console.log("PASS");
}
