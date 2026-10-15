/** Task 432 - Implement data structure for dynamic median (order-statistics tree). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Approach: two heaps with lazy deletion. Values up to the median live in a max-heap (low), the
 * rest in a min-heap (high); every live low value is <= every live high value. Deletions do not
 * search the heaps: a per-value pending count marks the removal and is consumed when a marked
 * value surfaces at a heap top (prune), so remove/insert are O(log n) amortised. median() returns
 * the lower median - the live element at 0-based index (size-1)//2 - which is exactly the top of
 * low, so even sizes answer with a stored value instead of a float; the empty multiset answers
 * null. Cross-checked against a sorted-list oracle (binary-search insertion) on LCG operation
 * scripts that insert, delete and re-query, including deletions until the structure empties.
 */
"use strict";

class Heap {
  constructor(less) {
    this.data = [];
    this.less = less;
  }

  size() {
    return this.data.length;
  }

  peek() {
    return this.data[0];
  }

  push(value) {
    const d = this.data;
    d.push(value);
    let i = d.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.less(d[i], d[parent])) break;
      [d[i], d[parent]] = [d[parent], d[i]];
      i = parent;
    }
  }

  pop() {
    const d = this.data;
    const top = d[0];
    const last = d.pop();
    if (d.length > 0) {
      d[0] = last;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let best = i;
        if (left < d.length && this.less(d[left], d[best])) best = left;
        if (right < d.length && this.less(d[right], d[best])) best = right;
        if (best === i) break;
        [d[i], d[best]] = [d[best], d[i]];
        i = best;
      }
    }
    return top;
  }
}

class DynamicMedian {
  constructor() {
    this.low = new Heap((a, b) => a > b); // max-heap: ceil(size/2) smallest live values
    this.high = new Heap((a, b) => a < b); // min-heap: the remaining live values
    this.pendingLow = new Map();
    this.pendingHigh = new Map();
    this.counts = new Map();
    this.total = 0;
    this.lowCount = 0;
  }

  size() {
    return this.total;
  }

  median() {
    if (this.total === 0) return null;
    this._prune();
    return this.low.peek();
  }

  insert(value) {
    this._prune();
    this.counts.set(value, (this.counts.get(value) || 0) + 1);
    if (this.low.size() > 0 && value > this.low.peek()) {
      this.high.push(value);
    } else {
      this.low.push(value);
      this.lowCount++;
    }
    this.total++;
    this._rebalance();
  }

  remove(value) {
    const live = this.counts.get(value) || 0;
    if (live === 0) return false;
    if (live === 1) this.counts.delete(value);
    else this.counts.set(value, live - 1);
    this._prune();
    if (this.low.size() > 0 && value <= this.low.peek()) {
      this.pendingLow.set(value, (this.pendingLow.get(value) || 0) + 1);
      this.lowCount--;
    } else {
      this.pendingHigh.set(value, (this.pendingHigh.get(value) || 0) + 1);
    }
    this.total--;
    this._rebalance();
    return true;
  }

  _prune() {
    while (this.low.size() > 0 && (this.pendingLow.get(this.low.peek()) || 0) > 0) {
      const top = this.low.pop();
      this.pendingLow.set(top, this.pendingLow.get(top) - 1);
    }
    while (this.high.size() > 0 && (this.pendingHigh.get(this.high.peek()) || 0) > 0) {
      const top = this.high.pop();
      this.pendingHigh.set(top, this.pendingHigh.get(top) - 1);
    }
  }

  _rebalance() {
    const target = Math.floor((this.total + 1) / 2);
    for (;;) {
      this._prune();
      if (this.lowCount > target) {
        const top = this.low.pop();
        this.lowCount--;
        this.high.push(top);
        continue;
      }
      if (this.lowCount < target) {
        const top = this.high.pop();
        this.lowCount++;
        this.low.push(top);
        continue;
      }
      break;
    }
  }
}

function oracleMedian(sortedValues) {
  if (sortedValues.length === 0) return null;
  return sortedValues[Math.floor((sortedValues.length - 1) / 2)];
}

function sortedInsert(sortedValues, value) {
  let lo = 0;
  let hi = sortedValues.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedValues[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  sortedValues.splice(lo, 0, value);
}

module.exports = { Heap, DynamicMedian, oracleMedian, sortedInsert };

if (require.main === module) {
  const assert = require("node:assert");
  let tracker = new DynamicMedian();
  assert.strictEqual(tracker.median(), null);
  assert.strictEqual(tracker.remove(0), false);

  const inserts = [5, 3, 8, 9];
  const expected = [5, 3, 5, 5];
  for (let i = 0; i < inserts.length; i++) {
    tracker.insert(inserts[i]);
    assert.strictEqual(tracker.median(), expected[i]);
  }
  assert.strictEqual(tracker.size(), 4);
  assert.strictEqual(tracker.remove(5), true);
  assert.strictEqual(tracker.median(), 8);
  assert.strictEqual(tracker.remove(3), true);
  assert.strictEqual(tracker.median(), 8);
  assert.strictEqual(tracker.remove(9), true);
  assert.strictEqual(tracker.median(), 8);
  assert.strictEqual(tracker.remove(8), true);
  assert.strictEqual(tracker.median(), null);
  assert.strictEqual(tracker.size(), 0);
  assert.strictEqual(tracker.remove(8), false);
  tracker.insert(7);
  assert.strictEqual(tracker.median(), 7);
  assert.strictEqual(tracker.remove(7), true);
  assert.strictEqual(tracker.median(), null);

  for (let i = 0; i < 3; i++) {
    tracker.insert(3);
    assert.strictEqual(tracker.median(), 3);
  }
  assert.strictEqual(tracker.remove(3), true);
  assert.strictEqual(tracker.remove(3), true);
  assert.strictEqual(tracker.median(), 3);
  assert.strictEqual(tracker.remove(3), true);
  assert.strictEqual(tracker.median(), null);

  tracker = new DynamicMedian();
  let oracle = [];
  for (let value = 1; value <= 10; value++) {
    tracker.insert(value);
    sortedInsert(oracle, value);
  }
  for (const value of [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]) {
    assert.strictEqual(tracker.remove(value), true);
    oracle.splice(oracle.indexOf(value), 1);
    assert.strictEqual(tracker.median(), oracleMedian(oracle));
  }
  assert.strictEqual(tracker.median(), null);

  tracker = new DynamicMedian();
  oracle = [];
  let seed = 4320;
  for (let step = 0; step < 400; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 4;
    if (op <= 1 || oracle.length === 0) {
      seed = (seed * 75 + 74) % 65537;
      const value = (seed % 40) - 20;
      tracker.insert(value);
      sortedInsert(oracle, value);
    } else if (op === 2) {
      seed = (seed * 75 + 74) % 65537;
      const value = oracle.splice(seed % oracle.length, 1)[0];
      assert.strictEqual(tracker.remove(value), true, `removal ${step}`);
    } else {
      seed = (seed * 75 + 74) % 65537;
      const missing = 100 + (seed % 10);
      assert.strictEqual(tracker.remove(missing), false, `absent ${step}`);
    }
    assert.strictEqual(tracker.median(), oracleMedian(oracle), `median step ${step}`);
    assert.strictEqual(tracker.size(), oracle.length, `size step ${step}`);
  }
  assert.ok(tracker.size() > 0);
  console.log("PASS");
}
