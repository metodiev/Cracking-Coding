/** Task 423 - Implement offline queries with divide & conquer on value (CDQ). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function countDominatingPairs(points) {
  const n = points.length;
  if (n < 2) return 0;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const sortedY = ys.slice().sort((a, b) => a - b);
  const ranks = new Map();
  for (const value of sortedY) {
    if (!ranks.has(value)) ranks.set(value, ranks.size);
  }
  const rank = ys.map((value) => ranks.get(value));
  const m = ranks.size;
  const tree = new Array(m + 1).fill(0);
  let total = 0;

  function bitAdd(r, delta) {
    let i = r + 1;
    while (i <= m) {
      tree[i] += delta;
      i += i & -i;
    }
  }

  function bitPrefix(count) {           // active values with rank < count
    let result = 0;
    let i = count;
    while (i > 0) {
      result += tree[i];
      i -= i & -i;
    }
    return result;
  }

  function cdq(lo, hi) {
    // returns indices [lo, hi) sorted by x descending; adds cross pairs to total
    if (hi - lo <= 1) return [lo];
    const mid = (lo + hi) >> 1;
    const left = cdq(lo, mid);
    const right = cdq(mid, hi);
    const touched = [];
    let pointer = 0;
    for (const j of right) {            // right is sorted by x descending too
      while (pointer < left.length && xs[left[pointer]] > xs[j]) {
        const r = rank[left[pointer]];
        bitAdd(r, 1);
        touched.push(r);
        pointer++;
      }
      total += pointer - bitPrefix(rank[j] + 1);
    }
    for (const r of touched) bitAdd(r, -1);
    const merged = [];
    let i = 0;
    let j = 0;
    while (i < left.length && j < right.length) {
      if (xs[left[i]] >= xs[right[j]]) merged.push(left[i++]);
      else merged.push(right[j++]);
    }
    while (i < left.length) merged.push(left[i++]);
    while (j < right.length) merged.push(right[j++]);
    return merged;
  }

  cdq(0, n);
  return total;
}

function bruteCount(points) {
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      if (points[i][0] > points[j][0] && points[i][1] > points[j][1]) total++;
    }
  }
  return total;
}

module.exports = { countDominatingPairs, bruteCount };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(countDominatingPairs([]), 0);
  assert.strictEqual(countDominatingPairs([[1, 1]]), 0);
  assert.strictEqual(countDominatingPairs([[3, 1], [2, 2]]), 0);
  assert.strictEqual(countDominatingPairs([[2, 2], [3, 1]]), 0);
  assert.strictEqual(countDominatingPairs([[5, 5], [3, 4], [1, 2]]), 3);
  assert.strictEqual(countDominatingPairs([[1, 1], [2, 2], [3, 3]]), 0);
  assert.strictEqual(countDominatingPairs([[3, 3], [3, 2]]), 0);   // equal x is not ">"
  assert.strictEqual(countDominatingPairs([[4, 4], [3, 4]]), 0);   // equal y is not ">"
  const decreasing = [];
  for (let i = 0; i < 5; i++) decreasing.push([7 - i, 9 - i]);
  assert.strictEqual(countDominatingPairs(decreasing), 10);        // all C(5,2) pairs count
  assert.strictEqual(countDominatingPairs([[5, 1], [4, 4], [6, 2], [3, 3]]), 1);

  let local = 424242;
  for (let trial = 0; trial < 40; trial++) {
    local = (local * 75 + 74) % 65537;
    const n = 2 + (local % 59);
    const points = [];
    for (let i = 0; i < n; i++) {
      local = (local * 75 + 74) % 65537;
      const x = (local % 11) - 5;
      local = (local * 75 + 74) % 65537;
      points.push([x, (local % 11) - 5]);
    }
    const got = countDominatingPairs(points);
    const expected = bruteCount(points);
    assert.strictEqual(got, expected, `trial ${trial} n=${n}`);
  }

  const equal = [];
  for (let i = 0; i < 12; i++) equal.push([7, 7]);
  assert.strictEqual(countDominatingPairs(equal), 0);

  console.log("PASS");
}
