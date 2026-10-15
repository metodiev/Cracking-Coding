/** Task 061 - Reverse singly linked list (iterative). (Tier: Beginner, CTCI Ch 2) */
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

function reverse(head) {
  let prev = null;
  while (head) {
    const next = head.next;
    head.next = prev;
    prev = head;
    head = next;
  }
  return prev;
}

module.exports = { ListNode, fromList, toList, reverse };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(toList(reverse(fromList([1, 2, 3]))), [3, 2, 1]);
  assert.deepStrictEqual(toList(reverse(fromList([1]))), [1]);
  assert.deepStrictEqual(toList(reverse(null)), []);
  console.log("PASS");
}
