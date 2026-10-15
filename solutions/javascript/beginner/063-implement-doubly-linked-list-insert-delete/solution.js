/** Task 063 - Implement doubly linked list insert/delete. (Tier: Beginner, CTCI Ch 2) */
"use strict";

class DNode {
  constructor(val) {
    this.val = val;
    this.prev = null;
    this.next = null;
  }
}

class DoublyLinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
  }

  append(val) {
    const node = new DNode(val);
    if (!this.tail) {
      this.head = this.tail = node;
    } else {
      node.prev = this.tail;
      this.tail.next = node;
      this.tail = node;
    }
    return node;
  }

  insertBefore(node, val) {
    const fresh = new DNode(val);
    fresh.prev = node.prev;
    fresh.next = node;
    if (node.prev) node.prev.next = fresh;
    else this.head = fresh;
    node.prev = fresh;
    return fresh;
  }

  delete(node) {
    if (node.prev) node.prev.next = node.next;
    else this.head = node.next;
    if (node.next) node.next.prev = node.prev;
    else this.tail = node.prev;
    node.prev = node.next = null;
  }

  toList() {
    const out = [];
    for (let p = this.head; p; p = p.next) out.push(p.val);
    return out;
  }

  toListReversed() {
    const out = [];
    for (let p = this.tail; p; p = p.prev) out.push(p.val);
    return out;
  }
}

module.exports = { DNode, DoublyLinkedList };

if (require.main === module) {
  const assert = require("node:assert");
  const d = new DoublyLinkedList();
  const a = d.append(1), b = d.append(2), c = d.append(3);
  assert.deepStrictEqual(d.toList(), [1, 2, 3]);
  assert.deepStrictEqual(d.toListReversed(), [3, 2, 1]);

  d.insertBefore(b, 9);
  assert.deepStrictEqual(d.toList(), [1, 9, 2, 3]);
  d.insertBefore(a, 0);
  assert.deepStrictEqual(d.toList(), [0, 1, 9, 2, 3]);
  assert.deepStrictEqual(d.toListReversed(), [3, 2, 9, 1, 0]);

  d.delete(a);
  assert.deepStrictEqual(d.toList(), [0, 9, 2, 3]);
  d.delete(c);
  assert.deepStrictEqual(d.toList(), [0, 9, 2]);
  d.delete(b);
  assert.deepStrictEqual(d.toList(), [0, 9]);
  assert.deepStrictEqual(d.toListReversed(), [9, 0]);
  console.log("PASS");
}
