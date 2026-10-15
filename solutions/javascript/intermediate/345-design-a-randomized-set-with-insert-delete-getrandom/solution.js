/**
 * Task 345 - RandomizedSet: insert/delete/getRandom in O(1). (Tier: Intermediate, CTCI Ch 7)
 *
 * Array + value -> index map; removal swaps the victim with the last element.  getRandom uses the
 * repo's deterministic LCG (seed = (seed*75+74) % 65537), so after the fixed operation script
 * every language produces the same draw sequence - asserted literally below.
 */
"use strict";

class RandomizedSet {
  constructor(seed = 2024) {
    this.items = [];
    this.index = new Map();
    this.rng = seed;
  }

  insert(value) {
    if (this.index.has(value)) return false;
    this.index.set(value, this.items.length);
    this.items.push(value);
    return true;
  }

  remove(value) {
    if (!this.index.has(value)) return false;
    const at = this.index.get(value);
    const last = this.items[this.items.length - 1];
    this.items[at] = last;               // move the last element into the hole
    this.index.set(last, at);
    this.items.pop();
    this.index.delete(value);
    return true;
  }

  getRandom() {
    if (this.items.length === 0) throw new Error("getRandom on an empty set");
    this.rng = (this.rng * 75 + 74) % 65537;
    return this.items[this.rng % this.items.length];
  }

  size() {
    return this.items.length;
  }

  toList() {
    return [...this.items];
  }

  validate() {
    if (this.items.length !== this.index.size) return false;
    for (const [value, at] of this.index) {
      if (this.items[at] !== value) return false;
    }
    return true;
  }
}

module.exports = { RandomizedSet };

if (require.main === module) {
  const assert = require("node:assert");

  const set = new RandomizedSet(2024);
  assert.strictEqual(set.size(), 0);

  const script = [[5, true], [3, true], [9, true], [1, true], [7, true], [5, false]];
  for (const [value, expected] of script) assert.strictEqual(set.insert(value), expected, `insert ${value}`);
  assert.strictEqual(set.remove(3), true, "remove present");
  assert.strictEqual(set.remove(42), false, "remove missing returns false");
  assert.strictEqual(set.insert(3), true, "re-insert");
  assert.deepStrictEqual(set.toList(), [5, 7, 9, 1, 3], "swap-remove layout");
  assert.strictEqual(set.size(), 5);
  assert.ok(set.validate(), "array and map agree");

  // deterministic LCG draws - identical in every language
  const draws = [5, 1, 3, 3, 3, 1, 1, 1];
  for (let i = 0; i < draws.length; i++) {
    assert.strictEqual(set.getRandom(), draws[i], `draw ${i}`);
  }

  const empty = new RandomizedSet(1);
  assert.throws(() => empty.getRandom(), Error, "getRandom on an empty set throws");

  const bulk = new RandomizedSet(98765);
  for (let v = 1; v <= 50; v++) assert.strictEqual(bulk.insert(v), true, `bulk insert ${v}`);
  for (let v = 1; v <= 50; v++) assert.strictEqual(bulk.insert(v), false, `bulk duplicate ${v}`);
  for (let v = 2; v <= 50; v += 2) assert.strictEqual(bulk.remove(v), true, `bulk remove ${v}`);
  for (let v = 2; v <= 50; v += 2) assert.strictEqual(bulk.remove(v), false, `bulk re-remove ${v}`);
  assert.strictEqual(bulk.size(), 25);
  assert.deepStrictEqual(bulk.toList().slice().sort((a, b) => a - b), Array.from({ length: 25 }, (_, i) => 2 * i + 1));
  assert.ok(bulk.validate(), "bulk layout consistent");
  const odds = new Set(Array.from({ length: 25 }, (_, i) => 2 * i + 1));
  const drawn = new Set();
  for (let i = 0; i < 500; i++) {
    const value = bulk.getRandom();
    assert.ok(odds.has(value), "draw is one of the odds");
    drawn.add(value);
  }
  assert.deepStrictEqual([...drawn].sort((a, b) => a - b), [...odds].sort((a, b) => a - b), "500 draws cover all odds");

  const small = new RandomizedSet(7);
  small.insert(10);
  small.insert(20);
  small.insert(30);
  assert.strictEqual(small.remove(10), true, "swap victim removed");
  assert.deepStrictEqual(small.toList(), [30, 20], "layout after victim removal");
  assert.strictEqual(small.remove(20), true, "last element removed");
  assert.deepStrictEqual(small.toList(), [30], "layout after last removal");
  assert.strictEqual(small.remove(30), true, "final removal");
  assert.deepStrictEqual(small.toList(), [], "empty again");
  assert.ok(small.validate(), "small layout consistent");
  console.log("PASS");
}
