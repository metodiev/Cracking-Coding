/** Task 339 - Range minimum query with offline queries (Mo's algorithm simple). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

/**
 * Simplified Mo's algorithm: block-sort the offline queries by (l / B, r) with B = n/sqrt(q)
 * so that sweeping the window boundaries performs only O((n + q) * sqrt(n)) add/remove moves
 * (the classic Mo bound), each O(log n) amortised on a min-heap multiset with lazy deletion.
 * Answers every range minimum in query order, cross-checked against a naive scan.
 */

class MinHeap {
  constructor() {
    this.data = [];
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
      if (this.data[parent] <= this.data[i]) break;
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
        if (left < this.data.length && this.data[left] < this.data[best]) best = left;
        if (right < this.data.length && this.data[right] < this.data[best]) best = right;
        if (best === i) break;
        [this.data[best], this.data[i]] = [this.data[i], this.data[best]];
        i = best;
      }
    }
    return top;
  }
}

class LazyMinMultiset {
  constructor() {
    this.heap = new MinHeap();
    this.pending = new Map();
    this.size = 0;
  }

  add(value) {
    this.heap.push(value);
    this.size += 1;
  }

  remove(value) {
    this.pending.set(value, (this.pending.get(value) || 0) + 1);
    this.size -= 1;
  }

  minimum() {
    while (this.heap.size() > 0 && (this.pending.get(this.heap.peek()) || 0) > 0) {
      const value = this.heap.pop();
      this.pending.set(value, this.pending.get(value) - 1);
    }
    if (this.size <= 0) throw new Error("minimum of an empty multiset");
    return this.heap.peek();
  }
}

function moRangeMin(values, queries) {
  const n = values.length;
  const q = queries.length;
  if (q === 0) return [];
  const root = Math.floor(Math.sqrt(q));
  const block = root > 0 ? Math.max(1, Math.floor(n / root)) : n;
  const order = Array.from({ length: q }, (_, i) => i).sort((a, b) => {
    const blockA = Math.floor(queries[a][0] / block);
    const blockB = Math.floor(queries[b][0] / block);
    if (blockA !== blockB) return blockA - blockB;
    const rightA = blockA % 2 === 0 ? queries[a][1] : -queries[a][1];
    const rightB = blockB % 2 === 0 ? queries[b][1] : -queries[b][1];
    return rightA - rightB;
  });

  const window = new LazyMinMultiset();
  const answers = new Array(q).fill(0);
  let currentLeft = 0, currentRight = -1;
  for (const index of order) {
    const [left, right] = queries[index];
    while (currentLeft > left) window.add(values[--currentLeft]);
    while (currentRight < right) window.add(values[++currentRight]);
    while (currentLeft < left) window.remove(values[currentLeft++]);
    while (currentRight > right) window.remove(values[currentRight--]);
    answers[index] = window.minimum();
  }
  return answers;
}

function naiveRangeMin(values, queries) {
  return queries.map(([left, right]) => Math.min(...values.slice(left, right + 1)));
}

module.exports = { moRangeMin, naiveRangeMin };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked minima on a small fixed array.
  const values = [1, 2, 1, 3, 2, 4];
  const queries = [[0, 2], [1, 4], [0, 5], [3, 3], [2, 5], [1, 1], [5, 5]];
  assert.deepStrictEqual(moRangeMin(values, queries), [1, 1, 1, 3, 1, 2, 4]);
  assert.deepStrictEqual(naiveRangeMin(values, queries), [1, 1, 1, 3, 1, 2, 4]);
  assert.deepStrictEqual(moRangeMin([7], [[0, 0]]), [7]);
  assert.deepStrictEqual(moRangeMin(values, []), []);

  // All-equal and strictly increasing arrays, including duplicate-heavy windows.
  assert.deepStrictEqual(moRangeMin(new Array(10).fill(5), [[0, 9], [2, 3], [4, 4]]), [5, 5, 5]);
  const rising = Array.from({ length: 10 }, (_, i) => i);
  assert.deepStrictEqual(moRangeMin(rising, [[0, 9], [0, 0], [3, 7]]), [0, 0, 3]);
  assert.deepStrictEqual(moRangeMin([2, 2, 3, 2, 3, 3], [[0, 5], [1, 4], [2, 5], [3, 5]]),
    [2, 2, 2, 2]);

  // LCG cross-check with a spread of sizes and query counts.
  let seed = 5150;
  for (let trial = 0; trial < 12; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 1 + (seed % 120);
    seed = (seed * 75 + 74) % 65537;
    const q = 1 + (seed % 200);
    seed = (seed * 75 + 74) % 65537;
    const alphabet = 1 + (seed % 12);
    const randomValues = [];
    for (let i = 0; i < n; i++) {
      seed = (seed * 75 + 74) % 65537;
      randomValues.push(seed % alphabet);
    }
    const randomQueries = [];
    for (let i = 0; i < q; i++) {
      seed = (seed * 75 + 74) % 65537;
      const left = seed % n;
      seed = (seed * 75 + 74) % 65537;
      randomQueries.push([left, left + (seed % (n - left))]);
    }
    assert.deepStrictEqual(moRangeMin(randomValues, randomQueries),
      naiveRangeMin(randomValues, randomQueries), `random ${trial}`);
  }

  // Single-position queries, one full query, and an adversarially reversed query order.
  seed = 424242;
  const singlesValues = [];
  for (let i = 0; i < 50; i++) {
    seed = (seed * 75 + 74) % 65537;
    singlesValues.push(seed % 7);
  }
  const singles = Array.from({ length: 50 }, (_, i) => [i, i]);
  assert.deepStrictEqual(moRangeMin(singlesValues, singles), singlesValues);
  assert.deepStrictEqual(moRangeMin(singlesValues, [[0, 49]]),
    naiveRangeMin(singlesValues, [[0, 49]]));
  assert.deepStrictEqual(moRangeMin(singlesValues, singles.slice().reverse()),
    naiveRangeMin(singlesValues, singles.slice().reverse()));
  console.log("PASS");
}
