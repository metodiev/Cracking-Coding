/** Task 468 - Implement streaming quantiles (Greenwald-Khanna outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Greenwald-Khanna ε-approximate quantile summary (SIGMOD 2001), batch merge-compress
 * variant: inserted values are buffered and every int(1/eps)+1 arrivals are merged
 * into the summary entries [value, g, delta] with band floor(2*eps*(n-1)); a fresh
 * value landing before an existing entry gets delta = entry.g + entry.delta - 1,
 * otherwise neighbours are absorbed while g_left + g_right + delta_right <= band.
 * QUANTILE(phi) accumulates g and returns the last entry with g_sum + delta within
 * int(phi*(n-1)+1) + int(eps*(n-1)), falling back to the tracked minimum.
 *
 * Checks against the exact sorted oracle: for every returned value v,
 *   rl = #{x < v} <= phi*N + eps*N   and   ru = #{x <= v} >= phi*N − eps*N,
 * with eps = 0.05, N = 2000, phi in {0, 0.05, ..., 1.0} over six stream patterns
 * (random, increasing, decreasing, constant, heavy duplicates, sawtooth); query(0)
 * returns the true minimum and the value is always a stream member. Documented
 * summary size bound (1/(2eps)+1)*(log2(2eps*n)+2) = 106 tuples, measured max 82.
 */
"use strict";

class GreenwaldKhanna {
  constructor(eps) {
    this.eps = eps;
    this.entries = []; // [value, g, delta]
    this.pending = [];
    this.count = 0;
    this.minimum = null;
  }

  get size() {
    return this.entries.length;
  }

  insert(value) {
    this.pending.push(value);
    this.count++;
    if (this.minimum === null || value < this.minimum) this.minimum = value;
    if (this.count % (Math.trunc(1.0 / this.eps) + 1) === 0) this.compress();
  }

  compress() {
    const threshold = Math.trunc(2.0 * this.eps * (this.count - 1));
    const fresh = this.pending.slice().sort((a, b) => a - b).map((v) => [v, 1, 0]);
    const old = this.entries;
    const out = [];
    let i = 0;
    let j = 0;
    const nf = fresh.length;
    const no = old.length;
    while (i < nf || j < no) {
      if (i === nf) {
        const t = old[j];
        j++;
        if (j < no && t[1] + old[j][1] + old[j][2] <= threshold) {
          old[j][1] += t[1];
          continue;
        }
        out.push(t);
      } else if (j === no) {
        const t = fresh[i];
        i++;
        if (i < nf && t[1] + fresh[i][1] + fresh[i][2] <= threshold) {
          fresh[i][1] += t[1];
          continue;
        }
        out.push(t);
      } else if (fresh[i][0] < old[j][0]) {
        const t = fresh[i];
        const u = old[j];
        if (t[1] + u[1] + u[2] <= threshold) {
          u[1] += t[1];
        } else {
          t[2] = u[1] + u[2] - t[1];
          out.push(t);
        }
        i++;
      } else {
        const t = old[j];
        j++;
        if (j < no && t[1] + old[j][1] + old[j][2] <= threshold) {
          old[j][1] += t[1];
          continue;
        }
        out.push(t);
      }
    }
    this.entries = out;
    this.pending = [];
  }

  query(phi) {
    if (this.count === 0) return null;
    if (this.pending.length > 0) this.compress();
    const rank = Math.trunc(phi * (this.count - 1) + 1);
    const spread = Math.trunc(this.eps * (this.count - 1));
    let total = 0;
    let i = 0;
    const n = this.entries.length;
    while (i < n) {
      total += this.entries[i][1];
      if (total + this.entries[i][2] > rank + spread) break;
      i++;
    }
    if (i === 0) return this.minimum;
    return this.entries[i - 1][0];
  }
}

function makeLcg(state) {
  let seed = state;
  return () => (seed = (seed * 75 + 74) % 65537);
}

function streams() {
  const lcg = makeLcg(20260116);
  const randomStream = [];
  for (let i = 0; i < 2000; i++) randomStream.push(lcg() % 1000);
  const dupes = [];
  for (let i = 0; i < 2000; i++) dupes.push(lcg() % 5);
  const increasing = Array.from({ length: 2000 }, (_, i) => i);
  const decreasing = Array.from({ length: 2000 }, (_, i) => 1999 - i);
  const constant = new Array(2000).fill(42);
  const sawtooth = Array.from({ length: 2000 }, (_, i) => i % 50);
  return [
    ["random", randomStream],
    ["increasing", increasing],
    ["decreasing", decreasing],
    ["constant", constant],
    ["dupes", dupes],
    ["sawtooth", sawtooth],
  ];
}

module.exports = { GreenwaldKhanna };

if (require.main === module) {
  const assert = require("node:assert");
  const EPS = 0.05;
  let maxSize = 0;
  for (const [name, stream] of streams()) {
    const n = stream.length;
    const sketch = new GreenwaldKhanna(EPS);
    for (const value of stream) sketch.insert(value);
    const exactly = stream.slice().sort((a, b) => a - b);
    for (let q = 0; q <= 20; q++) {
      const phi = q / 20.0;
      const value = sketch.query(phi);
      assert.ok(stream.includes(value), `${name} member ${phi}`);
      const lo = lowerBound(exactly, value);
      const hi = upperBound(exactly, value);
      const target = phi * n;
      const slack = EPS * n;
      assert.ok(lo <= target + slack + 1e-9, `${name} upper rank ${phi}`);
      assert.ok(hi >= target - slack - 1e-9, `${name} lower rank ${phi}`);
    }
    assert.strictEqual(sketch.query(0.0), Math.min(...stream), `${name} min`);
    maxSize = Math.max(maxSize, sketch.size); // after queries force compression
  }
  const bound = (1.0 / (2 * EPS) + 1) * (Math.log2(2 * EPS * 2000) + 2);
  assert.ok(maxSize <= bound, `size bound ${maxSize}`);
  assert.strictEqual(maxSize, 82); // measured maximum compressed size

  const empty = new GreenwaldKhanna(0.01);
  assert.strictEqual(empty.query(0.5), null);
  const tiny = new GreenwaldKhanna(0.01);
  for (const value of [5, 1, 9, 3, 7]) tiny.insert(value);
  assert.strictEqual(tiny.query(0.0), 1);
  assert.strictEqual(tiny.query(1.0), 9);
  assert.ok([3, 5].includes(tiny.query(0.5)));
  console.log("PASS");
}

function lowerBound(sorted, value) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(sorted, value) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] <= value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
