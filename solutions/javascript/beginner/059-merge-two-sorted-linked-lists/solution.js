/** Task 059 - Merge two sorted linked lists. (Tier: Beginner, CTCI Ch 2) */
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

function mergeSorted(a, b) {
  const dummy = new ListNode(0);
  let tail = dummy;
  while (a && b) {
    if (a.val <= b.val) { tail.next = a; a = a.next; }
    else                { tail.next = b; b = b.next; }
    tail = tail.next;
  }
  tail.next = a || b;
  return dummy.next;
}

module.exports = { ListNode, fromList, toList, mergeSorted };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(toList(mergeSorted(fromList([1, 3, 5]), fromList([2, 4, 6]))), [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(toList(mergeSorted(null, fromList([1]))), [1]);
  console.log("PASS");
}
