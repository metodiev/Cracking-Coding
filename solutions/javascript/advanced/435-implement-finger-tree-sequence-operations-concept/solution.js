/** Task 435 - Implement finger tree (sequence operations) concept. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Concept version of a finger tree: a sequence annotated by a monoid so that every subtree caches
 * the annotation of the elements below it, which is what lets a real finger tree answer prefix and
 * range measures in O(log n). The annotation monoid used here is M = (count, sum) with identity
 * (0, 0) and componentwise addition; measure() returns the cached annotation of the whole sequence
 * and measureRange(lo, hi) folds it over a range. The spine operations pushFront / pushBack /
 * popFront / popBack / front / back / concat / splitAt are simulated over a plain array while
 * keeping the annotation exact - documented honest O(n) internals, no worst-case finger-tree bound
 * claimed - and every scripted operation is asserted against a plain-list oracle (values, size,
 * cached measure and the monoid law combine(measure(a), measure(b)) == measure(concat(a, b))).
 */
"use strict";

const IDENTITY = [0, 0];

function combine(left, right) {
  return [left[0] + right[0], left[1] + right[1]];
}

class FingerTree {
  constructor(items = []) {
    this.items = items.slice();
    this.cached = IDENTITY.slice();
    for (const value of this.items) this.cached = combine(this.cached, [1, value]);
  }

  size() {
    return this.items.length;
  }

  measure() {
    return this.cached.slice();
  }

  measureRange(lo, hi) {
    if (lo < 0 || hi > this.items.length || lo > hi) throw new RangeError("range out of bounds");
    let total = IDENTITY.slice();
    for (let i = lo; i < hi; i++) total = combine(total, [1, this.items[i]]);
    return total;
  }

  front() {
    return this.items.length === 0 ? null : this.items[0];
  }

  back() {
    return this.items.length === 0 ? null : this.items[this.items.length - 1];
  }

  pushFront(value) {
    this.items.unshift(value);
    this.cached = combine([1, value], this.cached);
    return this;
  }

  pushBack(value) {
    this.items.push(value);
    this.cached = combine(this.cached, [1, value]);
    return this;
  }

  popFront() {
    if (this.items.length === 0) return null;
    const value = this.items.shift();
    this.cached = combine([-1, -value], this.cached);
    return value;
  }

  popBack() {
    if (this.items.length === 0) return null;
    const value = this.items.pop();
    this.cached = combine([-1, -value], this.cached);
    return value;
  }

  concat(other) {
    const joined = new FingerTree(this.items);
    joined.items = joined.items.concat(other.items);
    joined.cached = combine(this.cached, other.cached);
    return joined;
  }

  splitAt(index) {
    if (index < 0 || index > this.items.length) throw new RangeError("split index out of bounds");
    return [new FingerTree(this.items.slice(0, index)), new FingerTree(this.items.slice(index))];
  }

  toList() {
    return this.items.slice();
  }
}

module.exports = { IDENTITY, combine, FingerTree };

if (require.main === module) {
  const assert = require("node:assert");
  const empty = new FingerTree();
  assert.strictEqual(empty.size(), 0);
  assert.deepStrictEqual(empty.measure(), [0, 0]);
  assert.deepStrictEqual(empty.toList(), []);
  assert.strictEqual(empty.front(), null);
  assert.strictEqual(empty.back(), null);
  assert.strictEqual(empty.popFront(), null);
  assert.strictEqual(empty.popBack(), null);

  let tree = new FingerTree();
  tree.pushBack(1).pushBack(2).pushBack(3).pushFront(0);
  assert.deepStrictEqual(tree.toList(), [0, 1, 2, 3]);
  assert.strictEqual(tree.size(), 4);
  assert.deepStrictEqual(tree.measure(), [4, 6]);
  assert.strictEqual(tree.front(), 0);
  assert.strictEqual(tree.back(), 3);

  let parts = tree.splitAt(0);
  assert.deepStrictEqual(parts[0].toList(), []);
  assert.deepStrictEqual(parts[0].measure(), [0, 0]);
  assert.deepStrictEqual(parts[1].toList(), [0, 1, 2, 3]);
  parts = tree.splitAt(4);
  assert.deepStrictEqual(parts[0].toList(), [0, 1, 2, 3]);
  assert.deepStrictEqual(parts[1].measure(), [0, 0]);
  parts = tree.splitAt(2);
  assert.deepStrictEqual(parts[0].toList(), [0, 1]);
  assert.deepStrictEqual(parts[1].toList(), [2, 3]);
  assert.deepStrictEqual(combine(parts[0].measure(), parts[1].measure()), tree.measure());
  assert.deepStrictEqual(tree.measureRange(1, 3), [2, 3]);
  assert.deepStrictEqual(tree.measureRange(2, 2), [0, 0]);

  assert.deepStrictEqual(new FingerTree([9]).concat(new FingerTree()).toList(), [9]);
  assert.deepStrictEqual(new FingerTree().concat(new FingerTree([7])).toList(), [7]);
  assert.deepStrictEqual(new FingerTree([1, 2]).concat(new FingerTree([3])).measure(), [3, 6]);
  assert.deepStrictEqual(tree.concat(tree).toList(), [0, 1, 2, 3, 0, 1, 2, 3]);

  tree = new FingerTree([5, 6, 7]);
  assert.strictEqual(tree.popFront(), 5);
  assert.strictEqual(tree.popBack(), 7);
  assert.deepStrictEqual(tree.toList(), [6]);
  assert.deepStrictEqual(tree.measure(), [1, 6]);
  assert.strictEqual(tree.popFront(), 6);
  assert.strictEqual(tree.popFront(), null);
  assert.deepStrictEqual(tree.toList(), []);
  assert.deepStrictEqual(tree.measure(), [0, 0]);

  tree = new FingerTree();
  let oracle = [];
  let seed = 4350;
  for (let step = 0; step < 400; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 5;
    if (op === 0) {
      seed = (seed * 75 + 74) % 65537;
      const value = (seed % 50) - 25;
      tree.pushBack(value);
      oracle.push(value);
    } else if (op === 1) {
      seed = (seed * 75 + 74) % 65537;
      const value = (seed % 50) - 25;
      tree.pushFront(value);
      oracle.unshift(value);
    } else if (op === 2) {
      assert.strictEqual(tree.popFront(), oracle.length > 0 ? oracle.shift() : null, `pop front ${step}`);
    } else if (op === 3) {
      assert.strictEqual(tree.popBack(), oracle.length > 0 ? oracle.pop() : null, `pop back ${step}`);
    } else {
      seed = (seed * 75 + 74) % 65537;
      const index = seed % (oracle.length + 1);
      const split = tree.splitAt(index);
      assert.deepStrictEqual(split[0].toList(), oracle.slice(0, index), `split left ${step}`);
      assert.deepStrictEqual(split[1].toList(), oracle.slice(index), `split right ${step}`);
      assert.deepStrictEqual(combine(split[0].measure(), split[1].measure()), tree.measure(), `split combine ${step}`);
      tree = split[0].concat(split[1]);
    }
    assert.deepStrictEqual(tree.toList(), oracle, `list ${step}`);
    assert.strictEqual(tree.size(), oracle.length, `size ${step}`);
    assert.deepStrictEqual(tree.measure(), [oracle.length, oracle.reduce((a, b) => a + b, 0)], `measure ${step}`);
    if (oracle.length > 0) {
      assert.strictEqual(tree.front(), oracle[0], `front ${step}`);
      assert.strictEqual(tree.back(), oracle[oracle.length - 1], `back ${step}`);
    }
    seed = (seed * 75 + 74) % 65537;
    const lo = seed % (oracle.length + 1);
    seed = (seed * 75 + 74) % 65537;
    const hi = lo + (seed % (oracle.length - lo + 1));
    const rangeSum = oracle.slice(lo, hi).reduce((a, b) => a + b, 0);
    assert.deepStrictEqual(tree.measureRange(lo, hi), [hi - lo, rangeSum], `range ${step}`);
  }
  assert.ok(oracle.length > 0);
  console.log("PASS");
}
