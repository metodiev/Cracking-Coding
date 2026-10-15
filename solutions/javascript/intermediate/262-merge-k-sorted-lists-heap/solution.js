/** Task 262 - Merge K sorted lists (heap). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

class ListNode {
  constructor(val, next = null) {
    this.val = val;
    this.next = next;
  }
}

/** Generic binary min-heap driven by a comparator over the entries. */
class MinHeap {
  constructor(cmp) {
    this.cmp = cmp;
    this.data = [];
  }

  push(value) {
    this.data.push(value);
    this.siftUp(this.data.length - 1);
  }

  pop() {
    if (this.data.length === 0) throw new Error("heap empty");
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  siftUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.cmp(this.data[parent], this.data[i]) <= 0) break;
      [this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
      i = parent;
    }
  }

  siftDown(i) {
    const n = this.data.length;
    for (;;) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.cmp(this.data[l], this.data[smallest]) < 0) smallest = l;
      if (r < n && this.cmp(this.data[r], this.data[smallest]) < 0) smallest = r;
      if (smallest === i) break;
      [this.data[i], this.data[smallest]] = [this.data[smallest], this.data[i]];
      i = smallest;
    }
  }

  get size() { return this.data.length; }
}

function buildList(values) {
  let head = null;
  for (let i = values.length - 1; i >= 0; i--) head = new ListNode(values[i], head);
  return head;
}

function toArray(head) {
  const out = [];
  for (let p = head; p; p = p.next) out.push(p.val);
  return out;
}

/**
 * Min-heap over the current heads: each of the n nodes enters the heap once, so
 * O(n log k) time / O(k) space; nodes are reused, tails rewired in place. Returns
 * null for k = 0 and the SAME head object when k = 1. Entries are
 * [value, serial, node]; the serial breaks value ties deterministically.
 */
function mergeKLists(lists) {
  const heap = new MinHeap((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  let serial = 0;
  for (const node of lists) if (node) heap.push([node.val, serial++, node]);
  const dummy = new ListNode(0);
  let tail = dummy;
  while (heap.size > 0) {
    const node = heap.pop()[2];
    tail.next = node;
    tail = node;
    if (node.next) heap.push([node.next.val, serial++, node.next]);
  }
  tail.next = null;
  return dummy.next;
}

/** Flat sort reference used only for cross-checks; call BEFORE merging (merging rewires tails). */
function mergeOracle(lists) {
  const values = [];
  for (const node of lists) values.push(...toArray(node));
  values.sort((a, b) => a - b);
  return buildList(values);
}

module.exports = { ListNode, MinHeap, buildList, toArray, mergeKLists, mergeOracle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(toArray(mergeKLists([buildList([1, 4, 5]), buildList([1, 3, 4]), buildList([2, 6])])),
    [1, 1, 2, 3, 4, 4, 5, 6]);
  assert.strictEqual(mergeKLists([]), null);                          // k = 0 -> empty
  assert.strictEqual(mergeKLists([null, null]), null);                // only empty lists

  const single = buildList([1, 2, 3]);
  const same = mergeKLists([single]);
  assert.strictEqual(same, single, "k = 1 returns the same list");    // identity, not a copy
  assert.deepStrictEqual(toArray(same), [1, 2, 3]);

  assert.deepStrictEqual(toArray(mergeKLists([buildList([1]), buildList([2, 3, 4, 5]), buildList([6])])),
    [1, 2, 3, 4, 5, 6]);                                              // unequal lengths
  assert.deepStrictEqual(toArray(mergeKLists([buildList([1, 1]), buildList([1, 1])])), [1, 1, 1, 1]);  // duplicates
  assert.deepStrictEqual(toArray(mergeKLists([null, buildList([1]), null, buildList([])])), [1]);      // empties mixed in
  assert.deepStrictEqual(toArray(mergeKLists([buildList([-3, -1]), buildList([-2, 0])])), [-3, -2, -1, 0]);
  assert.deepStrictEqual(toArray(mergeKLists([buildList([5, 5]), buildList([5]), buildList([])])), [5, 5, 5]);

  // deterministic LCG so the cross-check is reproducible
  let seed = 262;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {        // cross-check vs the flat-sort oracle
    const k = randInt(0, 5);
    const lists = [];
    for (let i = 0; i < k; i++) {
      const values = Array.from({ length: randInt(0, 5) }, () => randInt(-6, 6)).sort((a, b) => a - b);
      lists.push(buildList(values));
    }
    const expected = toArray(mergeOracle(lists)); // oracle first: merging rewires tails
    assert.deepStrictEqual(toArray(mergeKLists(lists)), expected, `random agreement #${test}`);
  }
  console.log("PASS");
}
