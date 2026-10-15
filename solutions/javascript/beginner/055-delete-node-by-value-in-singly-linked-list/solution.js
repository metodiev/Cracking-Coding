/** Task 055 - Delete node by value in singly linked list. (Tier: Beginner, CTCI Ch 2) */
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

function deleteValue(head, value) {
  const dummy = new ListNode(0, head);
  let prev = dummy;
  while (prev.next) {
    if (prev.next.val === value) {
      prev.next = prev.next.next;
      break;
    }
    prev = prev.next;
  }
  return dummy.next;
}

module.exports = { ListNode, fromList, toList, deleteValue };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(toList(deleteValue(fromList([1, 2, 3]), 2)), [1, 3]);
  assert.deepStrictEqual(toList(deleteValue(fromList([1, 2, 3]), 1)), [2, 3]);
  assert.deepStrictEqual(toList(deleteValue(fromList([1]), 1)), []);
  console.log("PASS");
}
