/** Task 056 - Find length of linked list. (Tier: Beginner, CTCI Ch 2) */
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

function length(head) {
  let count = 0;
  for (let p = head; p; p = p.next) count++;
  return count;
}

module.exports = { ListNode, fromList, toList, length };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(length(fromList([1, 2, 3, 4])), 4);
  assert.strictEqual(length(null), 0);
  console.log("PASS");
}
