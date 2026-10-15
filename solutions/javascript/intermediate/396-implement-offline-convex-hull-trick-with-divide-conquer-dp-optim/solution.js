/** Task 396 - Implement the offline convex hull trick with divide-and-conquer DP optimization. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const INF = Number.POSITIVE_INFINITY;

class MinHull {
  constructor() {
    this.ms = [];
    this.bs = [];
  }

  addLine(m, b) {
    if (this.ms.length && m === this.ms[this.ms.length - 1]) {
      if (b >= this.bs[this.bs.length - 1]) return;
      this.ms.pop();
      this.bs.pop();
    }
    while (this.ms.length >= 2) {
      const m1 = this.ms[this.ms.length - 2];
      const b1 = this.bs[this.bs.length - 2];
      const m2 = this.ms[this.ms.length - 1];
      const b2 = this.bs[this.bs.length - 1];
      if ((b - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m)) {
        this.ms.pop();
        this.bs.pop();
      } else break;
    }
    this.ms.push(m);
    this.bs.push(b);
  }

  query(x) {
    if (!this.ms.length) throw new Error("query on an empty hull");
    let lo = 0;
    let hi = this.ms.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.ms[mid] * x + this.bs[mid] <= this.ms[mid + 1] * x + this.bs[mid + 1]) hi = mid;
      else lo = mid + 1;
    }
    return this.ms[lo] * x + this.bs[lo];
  }
}

function partitionBruteforce(xs, k) {
  const n = xs.length;
  if (k > n) return INF;
  let previous = [0, ...new Array(n).fill(INF)];
  for (let t = 1; t <= k; t++) {
    const current = new Array(n + 1).fill(INF);
    for (let i = t; i <= n; i++) {
      for (let j = t - 1; j < i; j++) {
        if (previous[j] === INF) continue;
        const span = xs[i - 1] - xs[j];
        const cost = previous[j] + span * span;
        if (cost < current[i]) current[i] = cost;
      }
    }
    previous = current;
  }
  return previous[n];
}

function partitionOfflineCht(xs, k) {
  const n = xs.length;
  if (k > n) return INF;
  let previous = [0, ...new Array(n).fill(INF)];
  for (let t = 1; t <= k; t++) {
    const current = new Array(n + 1).fill(INF);
    const hull = new MinHull();
    for (let i = t; i <= n; i++) {
      const j = i - 1;
      if (previous[j] !== INF) hull.addLine(-2 * xs[j], previous[j] + xs[j] * xs[j]);
      const x = xs[i - 1];
      current[i] = x * x + hull.query(x);
    }
    previous = current;
  }
  return previous[n];
}

function partitionDivideConquer(xs, k) {
  const n = xs.length;
  if (k > n) return INF;
  let previous = [0, ...new Array(n).fill(INF)];
  for (let t = 1; t <= k; t++) {
    const current = new Array(n + 1).fill(INF);
    current[t] = previous[t - 1];
    if (t + 1 <= n) solveDc(previous, current, xs, t + 1, n, t - 1, n - 1);
    previous = current;
  }
  return previous[n];
}

function solveDc(previous, current, xs, l, r, optl, optr) {
  if (l > r) return;
  const mid = (l + r) >> 1;
  let best = INF;
  let bestJ = -1;
  const upper = Math.min(optr, mid - 1);
  for (let j = optl; j <= upper; j++) {
    if (previous[j] === INF) continue;
    const span = xs[mid - 1] - xs[j];
    const cost = previous[j] + span * span;
    if (cost < best) {
      best = cost;
      bestJ = j;
    }
  }
  current[mid] = best;
  if (bestJ === -1) bestJ = optl;
  solveDc(previous, current, xs, l, mid - 1, optl, bestJ);
  solveDc(previous, current, xs, mid + 1, r, bestJ, optr);
}

function lcgArrays(count, seed = 99, length = 40) {
  const arrays = [];
  for (let a = 0; a < count; a++) {
    const xs = [0];
    for (let i = 1; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      xs.push(xs[xs.length - 1] + 1 + (seed % 20));
    }
    arrays.push(xs);
  }
  return arrays;
}

module.exports = { MinHull, partitionBruteforce, partitionOfflineCht, partitionDivideConquer, lcgArrays };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(partitionBruteforce([1, 3, 5], 2), 4);
  assert.strictEqual(partitionBruteforce([1, 3, 5], 1), 16);
  assert.strictEqual(partitionBruteforce([1, 3, 5], 3), 0);
  assert.strictEqual(partitionBruteforce([1, 3], 2), 0);
  assert.strictEqual(partitionBruteforce([1, 3], 3), INF);
  assert.strictEqual(partitionBruteforce([1], 1), 0);

  assert.strictEqual(partitionOfflineCht([1, 3, 5], 2), 4);
  assert.strictEqual(partitionDivideConquer([1, 3, 5], 2), 4);

  const hull = new MinHull();
  for (const [m, b] of [[1, 0], [0, 0], [-1, 0]]) hull.addLine(m, b);   // slopes non-increasing
  const normalise = (value) => (value === 0 ? 0 : value);               // -0 and 0 compare strictly unequal
  for (let x = -6; x <= 6; x++) {
    assert.strictEqual(normalise(hull.query(x)), normalise(Math.min(x, -x, 0)), String(x));
  }

  for (const xs of lcgArrays(8)) {
    for (let k = 1; k <= 6; k++) {
      const brute = partitionBruteforce(xs, k);
      assert.strictEqual(partitionOfflineCht(xs, k), brute, `cht k=${k}`);
      assert.strictEqual(partitionDivideConquer(xs, k), brute, `dc k=${k}`);
    }
  }
  assert.throws(() => new MinHull().query(3), /empty hull/);
  console.log("PASS");
}
