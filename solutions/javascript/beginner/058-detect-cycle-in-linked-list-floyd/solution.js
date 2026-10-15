/** Task 058 - Detect cycle in linked list (Floyd). (Tier: Beginner, CTCI Ch 2) */
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

function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}

module.exports = { ListNode, fromList, toList, hasCycle };

if (require.main === module) {
  const assert = require("node:assert");
  const a = fromList([1, 2, 3, 4]);
  assert.strictEqual(hasCycle(a), false);
  a.next.next.next.next = a.next;
  assert.strictEqual(hasCycle(a), true);
  console.log("PASS");
}
