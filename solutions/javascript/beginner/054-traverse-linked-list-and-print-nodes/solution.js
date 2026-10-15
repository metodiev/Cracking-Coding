/** Task 054 - Traverse linked list and print nodes. (Tier: Beginner, CTCI Ch 2) */
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

function traverse(head) {
  return toList(head);
}

module.exports = { ListNode, fromList, toList, traverse };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(traverse(fromList([1, 2, 3])), [1, 2, 3]);
  assert.deepStrictEqual(traverse(null), []);
  console.log("PASS");
}
