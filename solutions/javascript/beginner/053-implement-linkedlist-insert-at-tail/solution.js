/** Task 053 - Implement LinkedList insert at tail. (Tier: Beginner, CTCI Ch 2) */
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

function insertTail(head, value) {
  const node = new ListNode(value);
  if (!head) return node;
  let p = head;
  while (p.next) p = p.next;
  p.next = node;
  return head;
}

module.exports = { ListNode, fromList, toList, insertTail };

if (require.main === module) {
  const assert = require("node:assert");
  let head = null;
  for (const v of [1, 2, 3]) head = insertTail(head, v);
  assert.deepStrictEqual(toList(head), [1, 2, 3]);
  assert.deepStrictEqual(toList(insertTail(null, 7)), [7]);
  console.log("PASS");
}
