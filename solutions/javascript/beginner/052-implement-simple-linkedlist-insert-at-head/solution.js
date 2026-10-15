/** Task 052 - Implement simple LinkedList insert at head. (Tier: Beginner, CTCI Ch 2) */
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

function insertHead(head, value) {
  return new ListNode(value, head);
}

module.exports = { ListNode, fromList, toList, insertHead };

if (require.main === module) {
  const assert = require("node:assert");
  let head = null;
  for (const v of [1, 2, 3]) head = insertHead(head, v);
  assert.deepStrictEqual(toList(head), [3, 2, 1]);
  assert.deepStrictEqual(toList(insertHead(null, 9)), [9]);
  console.log("PASS");
}
