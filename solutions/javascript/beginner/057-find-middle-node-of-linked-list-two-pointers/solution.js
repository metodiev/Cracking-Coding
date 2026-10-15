/** Task 057 - Find middle node of linked list (two pointers). (Tier: Beginner, CTCI Ch 2) */
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

function middle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  return slow;
}

module.exports = { ListNode, fromList, toList, middle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(middle(fromList([1, 2, 3, 4, 5])).val, 3);
  assert.strictEqual(middle(fromList([1, 2, 3, 4])).val, 3);
  assert.strictEqual(middle(null), null);
  console.log("PASS");
}
