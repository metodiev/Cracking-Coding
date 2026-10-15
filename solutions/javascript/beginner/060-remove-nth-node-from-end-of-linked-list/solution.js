/** Task 060 - Remove nth node from end of linked list. (Tier: Beginner, CTCI Ch 2) */
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

function removeNthFromEnd(head, n) {
  const dummy = new ListNode(0, head);
  let slow = dummy, fast = dummy;
  for (let i = 0; i <= n; i++) fast = fast.next;
  while (fast) { slow = slow.next; fast = fast.next; }
  slow.next = slow.next.next;
  return dummy.next;
}

module.exports = { ListNode, fromList, toList, removeNthFromEnd };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(toList(removeNthFromEnd(fromList([1, 2, 3, 4, 5]), 2)), [1, 2, 3, 5]);
  assert.deepStrictEqual(toList(removeNthFromEnd(fromList([1, 2, 3, 4, 5]), 5)), [2, 3, 4, 5]);
  assert.deepStrictEqual(toList(removeNthFromEnd(fromList([1]), 1)), []);
  console.log("PASS");
}
