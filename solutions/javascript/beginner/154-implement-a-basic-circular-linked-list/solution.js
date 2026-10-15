/** Task 154 - Implement a basic circular linked list. (Tier: Beginner, CTCI Ch 2) */
"use strict";

class Node {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

/** Singly linked circular list: the tail always points back at the head. */
class CircularLinkedList {
  constructor() {
    this.head = null;
    this.size = 0;
  }

  /** Append at the tail; the new tail points back at the head. */
  append(value) {
    const node = new Node(value);
    if (!this.head) {
      node.next = node;
      this.head = node;
    } else {
      const tail = this.tail();
      tail.next = node;
      node.next = this.head;
    }
    this.size++;
  }

  /** Insert before position `index` (0 <= index <= size); size appends. */
  insert(index, value) {
    if (index < 0 || index > this.size) throw new RangeError("insert index out of range");
    if (index === this.size) {
      this.append(value);
      return;
    }
    const node = new Node(value);
    if (index === 0) {
      const tail = this.tail();
      node.next = this.head;
      tail.next = node;
      this.head = node;
    } else {
      let prev = this.head;
      for (let i = 0; i < index - 1; i++) prev = prev.next;
      node.next = prev.next;
      prev.next = node;
    }
    this.size++;
  }

  /** Delete the first node holding `value`; returns true if one was removed. */
  delete(value) {
    if (!this.head) return false;
    if (this.head.value === value) {
      if (this.size === 1) {
        this.head = null;
      } else {
        const tail = this.tail();
        this.head = this.head.next;
        tail.next = this.head;
      }
      this.size--;
      return true;
    }
    let prev = this.head;
    while (prev.next !== this.head && prev.next.value !== value) prev = prev.next;
    if (prev.next === this.head) return false;
    prev.next = prev.next.next;
    this.size--;
    return true;
  }

  /** The node whose next is the head (null for an empty list). */
  tail() {
    if (!this.head) return null;
    let node = this.head;
    while (node.next !== this.head) node = node.next;
    return node;
  }

  /** One lap: exactly `size` steps starting at the head (terminates). */
  toList() {
    const out = [];
    let node = this.head;
    for (let i = 0; i < this.size; i++) {
      out.push(node.value);
      node = node.next;
    }
    return out;
  }
}

module.exports = { Node, CircularLinkedList };

if (require.main === module) {
  const assert = require("node:assert");
  const lst = new CircularLinkedList();
  assert.deepStrictEqual(lst.toList(), []);
  assert.strictEqual(lst.head, null);

  for (const v of [1, 2, 3]) lst.append(v);
  assert.deepStrictEqual(lst.toList(), [1, 2, 3]);
  assert.strictEqual(lst.size, 3);
  assert.strictEqual(lst.tail().next, lst.head);            // wrap-around
  assert.strictEqual(lst.head.next.next.next, lst.head);

  lst.insert(0, 0);
  assert.deepStrictEqual(lst.toList(), [0, 1, 2, 3]);
  lst.insert(2, 99);
  assert.deepStrictEqual(lst.toList(), [0, 1, 99, 2, 3]);
  lst.insert(lst.size, 4);
  assert.deepStrictEqual(lst.toList(), [0, 1, 99, 2, 3, 4]);
  assert.throws(() => lst.insert(99, 0), RangeError);

  assert.strictEqual(lst.delete(99), true);
  assert.deepStrictEqual(lst.toList(), [0, 1, 2, 3, 4]);
  assert.strictEqual(lst.delete(0), true);                  // deleting the head
  assert.strictEqual(lst.head.value, 1);
  assert.deepStrictEqual(lst.toList(), [1, 2, 3, 4]);
  assert.strictEqual(lst.tail().next, lst.head);
  assert.strictEqual(lst.delete(4), true);                  // delete the tail
  assert.deepStrictEqual(lst.toList(), [1, 2, 3]);
  assert.strictEqual(lst.delete(42), false);                // missing value

  assert.strictEqual(lst.delete(1) && lst.delete(2) && lst.delete(3), true);
  assert.deepStrictEqual(lst.toList(), []);                 // last node -> empty
  assert.strictEqual(lst.head, null);
  assert.strictEqual(lst.delete(1), false);
  lst.append(7);
  assert.deepStrictEqual(lst.toList(), [7]);
  assert.strictEqual(lst.head.next, lst.head);              // single node self-loop
  assert.strictEqual(lst.delete(7), true);
  assert.deepStrictEqual(lst.toList(), []);
  console.log("PASS");
}
