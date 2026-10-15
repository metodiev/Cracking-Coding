/** Task 251 - Implement skip list insert/search/delete. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const MAX_LEVEL = 6;

class SkipListNode {
  constructor(key, level) {
    this.key = key;
    this.level = level;
    this.next = new Array(level).fill(null);
  }
}

/**
 * Layered sorted list: level 0 holds every key, higher levels are sparse express lanes.
 * Heights come from a deterministic 31-bit LCG coin so every language builds the same
 * structure. Invariant: level i's chain is exactly the level-0 subsequence of nodes whose
 * height is greater than i.
 */
class SkipList {
  constructor(seed = 20251) {
    this.head = new SkipListNode(-1, MAX_LEVEL);
    this.size = 0;
    this.state = seed;
  }

  coin() {
    this.state = (Math.imul(this.state, 1103515245) + 12345) & 0x7fffffff;
    return (this.state >> 16) & 1;
  }

  randomLevel() {
    let level = 1;
    while (level < MAX_LEVEL && this.coin() === 1) level += 1;
    return level;
  }

  find(key) {
    const update = new Array(MAX_LEVEL);
    let node = this.head;
    for (let i = MAX_LEVEL - 1; i >= 0; i -= 1) {
      while (node.next[i] && node.next[i].key < key) node = node.next[i];
      update[i] = node;
    }
    return update;
  }

  insert(key) {
    const update = this.find(key);
    const after = update[0].next[0];
    if (after && after.key === key) return false;
    const node = new SkipListNode(key, this.randomLevel());
    for (let i = 0; i < node.level; i += 1) {
      node.next[i] = update[i].next[i];
      update[i].next[i] = node;
    }
    this.size += 1;
    return true;
  }

  search(key) {
    let node = this.head;
    for (let i = MAX_LEVEL - 1; i >= 0; i -= 1) {
      while (node.next[i] && node.next[i].key < key) node = node.next[i];
      if (node.next[i] && node.next[i].key === key) return true;
    }
    return false;
  }

  delete(key) {
    const update = this.find(key);
    const target = update[0].next[0];
    if (!target || target.key !== key) return false;
    for (let i = 0; i < target.level; i += 1) update[i].next[i] = target.next[i];
    this.size -= 1;
    return true;
  }

  walk(level = 0) {
    const out = [];
    for (let node = this.head.next[level]; node; node = node.next[level]) out.push(node.key);
    return out;
  }

  /** Heights of the level-0 nodes, in sorted walk order. */
  walkHeights() {
    const out = [];
    for (let node = this.head.next[0]; node; node = node.next[0]) out.push(node.level);
    return out;
  }
}

module.exports = { MAX_LEVEL, SkipListNode, SkipList };

if (require.main === module) {
  const assert = require("node:assert");
  const keys = [3, 6, 7, 9, 12, 19, 17, 26, 21, 25];
  const heights = [1, 1, 1, 1, 1, 4, 1, 6, 1, 4];      // seed 20251, sorted walk order

  const sl = new SkipList();
  for (const key of keys) assert.strictEqual(sl.insert(key), true);
  assert.strictEqual(sl.size, 10);
  assert.deepStrictEqual(sl.walkHeights(), heights);     // identical structure everywhere
  assert.strictEqual(Math.max(...heights), MAX_LEVEL);
  assert.deepStrictEqual(sl.walk(0), [...keys].sort((a, b) => a - b));

  for (const key of keys) assert.strictEqual(sl.search(key), true);
  for (const key of [-1, 0, 5, 10, 20, 22, 30]) assert.strictEqual(sl.search(key), false);

  const lv = sl.walkHeights();
  const walk0 = sl.walk(0);
  for (let i = 0; i < MAX_LEVEL; i += 1) {               // index consistency
    const expected = walk0.filter((_, j) => lv[j] > i);
    const chain = sl.walk(i);
    assert.deepStrictEqual(chain, expected);
    assert.deepStrictEqual(chain, [...chain].sort((a, b) => a - b));
  }

  assert.strictEqual(sl.insert(3), false);               // duplicates rejected
  assert.strictEqual(sl.size, 10);

  assert.strictEqual(sl.delete(19), true);
  assert.strictEqual(sl.size, 9);
  assert.strictEqual(sl.search(19), false);
  assert.strictEqual(sl.delete(19), false);
  assert.strictEqual(sl.delete(4), false);
  assert.strictEqual(sl.delete(100), false);
  const lv2 = sl.walkHeights();
  const walk02 = sl.walk(0);
  for (let i = 0; i < MAX_LEVEL; i += 1) {
    assert.deepStrictEqual(sl.walk(i), walk02.filter((_, j) => lv2[j] > i));
  }

  assert.strictEqual(sl.delete(3), true);                // head-adjacent
  assert.strictEqual(sl.delete(25), true);
  assert.strictEqual(sl.delete(21), true);               // top-lane node
  for (const key of [6, 7, 9, 12, 17, 26]) assert.strictEqual(sl.delete(key), true);
  assert.strictEqual(sl.size, 0);
  for (let i = 0; i < MAX_LEVEL; i += 1) assert.deepStrictEqual(sl.walk(i), []);
  for (const key of keys) assert.strictEqual(sl.search(key), false);

  assert.strictEqual(sl.insert(42), true);
  assert.strictEqual(sl.search(42), true);
  assert.strictEqual(sl.delete(42), true);
  assert.strictEqual(sl.size, 0);
  console.log("PASS");
}
