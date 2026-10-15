/** Task 334 - Implement rolling median with delete operation (multiset or two heaps + lazy deletion). (Tier: Intermediate, CTCI Ch 7) */
"use strict";

/**
 * MedianMultiset keeps the smaller half of the values in a max-heap (low) and the larger
 * half in a min-heap (high), so low.length == high.length or low.length == high.length + 1.
 * delete() marks one occurrence as pending in whichever heap holds it; the mark is consumed
 * when that value reaches a heap top. Cross-checked against a sorted-list oracle.
 */

class BinaryHeap {
  constructor(compare) {
    this.data = [];
    this.compare = compare;                 // negative when a belongs above b
  }

  size() {
    return this.data.length;
  }

  peek() {
    return this.data[0];
  }

  push(value) {
    this.data.push(value);
    let i = this.data.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(this.data[parent], this.data[i]) <= 0) break;
      [this.data[parent], this.data[i]] = [this.data[i], this.data[parent]];
      i = parent;
    }
  }

  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let best = i;
        if (left < this.data.length && this.compare(this.data[left], this.data[best]) < 0) best = left;
        if (right < this.data.length && this.compare(this.data[right], this.data[best]) < 0) best = right;
        if (best === i) break;
        [this.data[best], this.data[i]] = [this.data[i], this.data[best]];
        i = best;
      }
    }
    return top;
  }
}

class MedianMultiset {
  constructor() {
    this.low = new BinaryHeap((a, b) => b - a);
    this.high = new BinaryHeap((a, b) => a - b);
    this.pendingLow = new Map();
    this.pendingHigh = new Map();
    this.pendingLowTotal = 0;
    this.pendingHighTotal = 0;
    this.active = new Map();
    this.count = 0;
  }

  size() {
    return this.count;
  }

  add(value) {
    this.cleanLow();
    this.cleanHigh();
    if (this.low.size() > 0 && value <= this.low.peek()) this.low.push(value);
    else this.high.push(value);
    this.count += 1;
    this.active.set(value, (this.active.get(value) || 0) + 1);
    this.rebalance();
  }

  /** Removes one occurrence of `value`; throws when it is absent. */
  delete(value) {
    if ((this.active.get(value) || 0) < 1) throw new Error(`value not present: ${value}`);
    this.active.set(value, this.active.get(value) - 1);
    this.cleanLow();
    if (this.low.size() > 0 && value <= this.low.peek()) {
      this.pendingLow.set(value, (this.pendingLow.get(value) || 0) + 1);
      this.pendingLowTotal += 1;
    } else {
      this.pendingHigh.set(value, (this.pendingHigh.get(value) || 0) + 1);
      this.pendingHighTotal += 1;
    }
    this.count -= 1;
    this.cleanLow();
    this.cleanHigh();
    this.rebalance();
  }

  /** Median as a number, or null when the multiset is empty. */
  median() {
    this.cleanLow();
    this.cleanHigh();
    if (this.count === 0) return null;
    if (this.count % 2 === 1) return normalise(this.low.peek());
    return normalise((this.low.peek() + this.high.peek()) / 2);
  }

  cleanLow() {
    while (this.low.size() > 0 && (this.pendingLow.get(this.low.peek()) || 0) > 0) {
      const value = this.low.pop();
      this.pendingLow.set(value, this.pendingLow.get(value) - 1);
      this.pendingLowTotal -= 1;
    }
  }

  cleanHigh() {
    while (this.high.size() > 0 && (this.pendingHigh.get(this.high.peek()) || 0) > 0) {
      const value = this.high.pop();
      this.pendingHigh.set(value, this.pendingHigh.get(value) - 1);
      this.pendingHighTotal -= 1;
    }
  }

  rebalance() {
    for (;;) {
      const lowSize = this.low.size() - this.pendingLowTotal;
      const highSize = this.high.size() - this.pendingHighTotal;
      if (lowSize > highSize + 1) {
        this.cleanLow();
        this.high.push(this.low.pop());
      } else if (highSize > lowSize) {
        this.cleanHigh();
        this.low.push(this.high.pop());
      } else {
        return;
      }
    }
  }
}

/** Keeps a computed 0 as +0 so assert.strictEqual never trips over -0. */
function normalise(value) {
  return value === 0 ? 0 : value;
}

class SortedListOracle {
  constructor() {
    this.data = [];
  }

  add(value) {
    let index = 0;
    while (index < this.data.length && this.data[index] < value) index += 1;
    this.data.splice(index, 0, value);
  }

  delete(value) {
    const index = this.data.indexOf(value);
    if (index < 0) throw new Error("oracle delete of absent value");
    this.data.splice(index, 1);
  }

  median() {
    const n = this.data.length;
    if (n === 0) return null;
    if (n % 2 === 1) return normalise(this.data[n >> 1]);
    return normalise((this.data[(n >> 1) - 1] + this.data[n >> 1]) / 2);
  }
}

module.exports = { MedianMultiset, SortedListOracle };

if (require.main === module) {
  const assert = require("node:assert");

  const m = new MedianMultiset();
  assert.strictEqual(m.size(), 0);
  assert.strictEqual(m.median(), null);
  m.add(5);
  assert.strictEqual(m.median(), 5);
  m.add(3);
  assert.strictEqual(m.size(), 2);
  assert.strictEqual(m.median(), 4);
  m.add(8);
  assert.strictEqual(m.median(), 5);
  m.add(1);
  assert.strictEqual(m.median(), 4);              // {1,3,5,8}
  m.delete(5);
  assert.strictEqual(m.size(), 3);
  assert.strictEqual(m.median(), 3);              // {1,3,8}
  m.delete(1);
  assert.strictEqual(m.median(), 5.5);            // {3,8}
  m.delete(3);
  assert.strictEqual(m.median(), 8);              // {8}
  m.delete(8);
  assert.strictEqual(m.size(), 0);
  assert.strictEqual(m.median(), null);
  assert.throws(() => m.delete(7), /not present/);

  // Duplicates: every occurrence is tracked separately.
  const dupes = new MedianMultiset();
  for (let i = 0; i < 3; i++) dupes.add(4);
  assert.strictEqual(dupes.median(), 4);
  dupes.delete(4);
  assert.strictEqual(dupes.median(), 4);
  dupes.delete(4);
  assert.strictEqual(dupes.size(), 1);
  assert.strictEqual(dupes.median(), 4);
  dupes.delete(4);
  assert.strictEqual(dupes.median(), null);

  // Alternating deletes across the median boundary exercise lazy deletion on both heaps.
  const seq = new MedianMultiset();
  for (let value = 1; value <= 10; value++) seq.add(value);
  assert.strictEqual(seq.median(), 5.5);
  const steps = [[5, 6], [6, 5.5], [10, 4], [1, 5.5], [7, 4], [2, 6]];
  for (const [value, expected] of steps) {
    seq.delete(value);
    assert.strictEqual(seq.median(), expected, `after deleting ${value}`);
  }
  assert.strictEqual(seq.size(), 4);

  let seed = 4242;
  const fast = new MedianMultiset();
  const oracle = new SortedListOracle();
  for (let step = 0; step < 2000; step++) {
    seed = (seed * 75 + 74) % 65537;
    if (oracle.data.length > 0 && seed % 10 >= 6) {
      seed = (seed * 75 + 74) % 65537;
      const value = oracle.data[seed % oracle.data.length];
      fast.delete(value);
      oracle.delete(value);
    } else {
      seed = (seed * 75 + 74) % 65537;
      const value = 1 + (seed % 50);
      fast.add(value);
      oracle.add(value);
    }
    assert.strictEqual(fast.size(), oracle.data.length, `random size ${step}`);
    seed = (seed * 75 + 74) % 65537;
    if (seed % 5 === 0 || step % 7 === 0) {
      assert.strictEqual(fast.median(), oracle.median(), `random median ${step}`);
    }
  }
  while (oracle.data.length > 0) {
    const value = oracle.data[0];
    fast.delete(value);
    oracle.delete(value);
    assert.strictEqual(fast.median(), oracle.median(), "drain median");
  }
  assert.strictEqual(fast.median(), null);
  console.log("PASS");
}
