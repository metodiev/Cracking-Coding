/** Task 062 - Reverse singly linked list (recursive). (Tier: Beginner, CTCI Ch 2) */
"use strict";

class ListNode {
  constructor(val, next = null) {
    this.val = val;
    this.next = next;
  }
}

function fromList(values) {
  let head = null;
  for (let i = values.length - 1; i >= 0; i--) head = new ListNode(values[i], head);
  return head;
}

function toList(head) {
  const out = [];
  for (let p = head; p; p = p.next) out.push(p.val);
  return out;
}

function reverseRecursive(head) {
  if (!head || !head.next) return head;
  const newHead = reverseRecursive(head.next);
  head.next.next = head;
  head.next = null;
  return newHead;
}

module.exports = { ListNode, fromList, toList, reverseRecursive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(toList(reverseRecursive(fromList([1, 2, 3, 4]))), [4, 3, 2, 1]);
  assert.deepStrictEqual(toList(reverseRecursive(fromList([9]))), [9]);
  assert.deepStrictEqual(toList(reverseRecursive(null)), []);
  console.log("PASS");
}
