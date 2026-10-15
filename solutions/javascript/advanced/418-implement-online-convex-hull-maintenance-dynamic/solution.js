/** Task 418 - Implement online convex hull maintenance (dynamic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Online convex hull trick supporting BOTH signs and BOTH insertion orders:
 *   * mode 'max' maintains the upper envelope, mode 'min' the lower envelope,
 *   * order 'inc' inserts lines with non-decreasing slopes, order 'dec' with non-increasing
 *     ones — both are the orders produced by sorting candidate lines by slope.
 * The hull is kept as arrays (ms, bs) of lines whose breakpoints increase. When appending
 * (m, b) after (m1, b1), (m2, b2) the middle line m2 is unnecessary exactly when
 *     sign * [(b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1)] >= 0,
 * where sign = +1 for (max, inc) / (min, dec) and -1 for (max, dec) / (min, inc); the
 * condition compares the two intersection abscissas without ever dividing. Duplicate
 * slopes keep only the best intercept (larger for max, smaller for min). Queries are
 * unimodal over the hull, so an O(log n) adjacent-pair binary search finds the peak
 * (for max) or the valley (for min). Deletions are out of scope: this is add-only.
 * Slopes/values here stay well below 2^53, so plain Numbers are exact — no BigInt.
 */
"use strict";

class ConvexHullTrick {
  constructor(mode, order) {
    if ((mode !== "max" && mode !== "min") || (order !== "inc" && order !== "dec")) {
      throw new Error("mode must be max/min and order inc/dec");
    }
    this.forMax = mode === "max";
    this.sign = (mode === "max" ? 1 : -1) * (order === "inc" ? 1 : -1);
    this.ms = [];
    this.bs = [];
  }

  size() {
    return this.ms.length;
  }

  addLine(m, b) {
    if (this.ms.length > 0 && m === this.ms[this.ms.length - 1]) {
      const lastB = this.bs[this.bs.length - 1];
      const better = this.forMax ? b > lastB : b < lastB;
      if (!better) return;
      this.ms.pop();
      this.bs.pop();
    }
    while (this.ms.length >= 2) {
      const m1 = this.ms[this.ms.length - 2], b1 = this.bs[this.bs.length - 2];
      const m2 = this.ms[this.ms.length - 1], b2 = this.bs[this.bs.length - 1];
      const expr = (b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1);
      if (this.sign * expr >= 0) {
        this.ms.pop();
        this.bs.pop();
      } else {
        break;
      }
    }
    this.ms.push(m);
    this.bs.push(b);
  }

  query(x) {
    if (this.ms.length === 0) throw new Error("query on an empty hull");
    let lo = 0, hi = this.ms.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      const vMid = this.ms[mid] * x + this.bs[mid];
      const vNext = this.ms[mid + 1] * x + this.bs[mid + 1];
      if (this.forMax) {
        if (vMid <= vNext) lo = mid + 1;
        else hi = mid;
      } else {
        if (vMid >= vNext) lo = mid + 1;
        else hi = mid;
      }
    }
    return this.ms[lo] * x + this.bs[lo];
  }
}

let seed = 0;

function nextRand() {
  seed = (seed * 75 + 74) % 65537;
  return seed;
}

function bruteBest(lines, x, forMax) {
  let best = forMax ? -Infinity : Infinity;
  for (const [m, b] of lines) {
    const value = m * x + b;
    best = forMax ? Math.max(best, value) : Math.min(best, value);
  }
  return best;
}

function lcgLines(count) {
  const lines = [];
  for (let k = 0; k < count; k++) {
    lines.push([nextRand() % 141 - 70, nextRand() % 4001 - 2000]);
  }
  return lines;
}

function sortBySlope(lines, increasing) {
  lines.sort((p, q) => (increasing ? p[0] - q[0] : q[0] - p[0]));
}

module.exports = { ConvexHullTrick, bruteBest, lcgLines, sortBySlope, nextRand };

if (require.main === module) {
  const assert = require("node:assert");

  // hand case: lines y = x, y = 2x + 3, y = 4x + 5 (max): 5 at 0, 45 at 10, -10 at -10
  const hull = new ConvexHullTrick("max", "inc");
  for (const [m, b] of [[1, 0], [2, 3], [4, 5]]) hull.addLine(m, b);
  assert.strictEqual(hull.query(0), 5);
  assert.strictEqual(hull.query(10), 45);
  assert.strictEqual(hull.query(-10), -10);
  // the same lines inserted from the other end of the slope order agree everywhere
  const dec = new ConvexHullTrick("max", "dec");
  for (const [m, b] of [[4, 5], [2, 3], [1, 0]]) dec.addLine(m, b);
  for (let x = -10; x <= 10; x++) assert.strictEqual(dec.query(x), hull.query(x), `order agreement ${x}`);

  // hand case for min: y = 2x + 5, y = x + 1: 1 at 0, 11 at 10, -15 at -10
  const mn = new ConvexHullTrick("min", "inc");
  mn.addLine(2, 5);
  mn.addLine(1, 1);
  assert.strictEqual(mn.query(0), 1);
  assert.strictEqual(mn.query(10), 11);
  assert.strictEqual(mn.query(-10), -15);
  const mnDec = new ConvexHullTrick("min", "dec");
  mnDec.addLine(1, 1);
  mnDec.addLine(2, 5);
  for (let x = -10; x <= 10; x++) assert.strictEqual(mnDec.query(x), mn.query(x), `min order agreement ${x}`);

  // duplicate slopes keep only the best intercept: max keeps b = -1 (value 20 at x = 7)
  const dup = new ConvexHullTrick("max", "inc");
  for (let i = 0; i < 4; i++) dup.addLine(3, -1);
  assert.strictEqual(dup.query(7), 20);
  assert.strictEqual(dup.size(), 1);
  const dup2 = new ConvexHullTrick("min", "inc");
  dup2.addLine(3, 10);
  dup2.addLine(3, 5);
  assert.strictEqual(dup2.query(7), 26);
  assert.strictEqual(dup2.size(), 1);
  const dup3 = new ConvexHullTrick("max", "inc");
  dup3.addLine(3, -1);
  dup3.addLine(3, -2);
  assert.strictEqual(dup3.size(), 1);
  assert.strictEqual(dup3.query(7), 20);

  // fixed-line oracle in all four mode/order combinations
  const fixedSrc = [[1, 0], [-1, 3], [0, -2], [5, -10], [-4, 6]];
  for (const mode of ["max", "min"]) {
    for (const order of ["inc", "dec"]) {
      const fixed = fixedSrc.map((p) => p.slice());
      sortBySlope(fixed, order === "inc");
      const probe = new ConvexHullTrick(mode, order);
      for (const [m, b] of fixed) probe.addLine(m, b);
      for (let x = -20; x <= 20; x++) {
        assert.strictEqual(probe.query(x), bruteBest(fixed, x, mode === "max"), `fixed ${mode} ${order} ${x}`);
      }
    }
  }

  // interleaved add/query scripts from the deterministic LCG, all four combinations
  let index = 0;
  for (const mode of ["max", "min"]) {
    for (const order of ["inc", "dec"]) {
      seed = 4180 + index++;
      const raw = lcgLines(60);
      sortBySlope(raw, order === "inc");
      const h = new ConvexHullTrick(mode, order);
      const added = [];
      for (let k = 0; k < raw.length; k++) {
        h.addLine(raw[k][0], raw[k][1]);
        added.push(raw[k]);
        assert.ok(h.size() <= added.length, "hull size bound");
        for (let t = 0; t < h.size(); t++) {
          const present = added.some((line) => line[0] === h.ms[t] && line[1] === h.bs[t]);
          assert.ok(present, "hull line present");
        }
        if (k % 3 === 2 || k === raw.length - 1) {
          for (let q = 0; q < 3; q++) {
            const x = nextRand() % 2001 - 1000;
            assert.strictEqual(h.query(x), bruteBest(added, x, mode === "max"),
              `script query ${index} ${k} ${x}`);
          }
        }
      }
      for (let x = -200; x <= 200; x++) {
        assert.strictEqual(h.query(x), bruteBest(raw, x, mode === "max"), `sweep ${index} ${x}`);
      }
      const distinct = new Set(raw.map((line) => line[0]));
      assert.ok(h.size() <= distinct.size, "distinct slope bound");
    }
  }

  // querying before the first insertion is rejected
  assert.throws(() => new ConvexHullTrick("max", "inc").query(3), /empty hull/);

  console.log("PASS");
}
