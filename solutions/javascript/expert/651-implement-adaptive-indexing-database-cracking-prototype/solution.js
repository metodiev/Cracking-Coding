/** Task 651 - Implement adaptive indexing (database cracking) prototype. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Database cracking (Idreos et al.) builds the index *incrementally* as a side effect of
 * answering queries instead of paying one upfront sort. The column lives in one array
 * `data` partitioned into contiguous *pieces*; piece p owns slot range
 * [slots[p][0], slots[p][1]) and carries inclusive key bounds [bounds[p][0], bounds[p][1]].
 *
 * Invariant P (re-established after every query):
 *   (a) pieces partition the whole array and are ordered left to right,
 *   (b) every value inside a piece lies within that piece's bounds,
 *   (c) bounds are strictly increasing: bounds[p][1] < bounds[p+1][0],
 *   (d) the value multiset is unchanged (cracking only permutes in place).
 * (b)+(c) let a query decide "whole piece matches" or "cannot match" from bounds alone, so
 * only pieces straddling a query endpoint are physically cracked (three-way in-place
 * partition by low and by high). That is what makes the index adaptive: the piece count
 * grows monotonically with the workload and converges to sorted order, and each crack
 * shrinks the future work for the same region.
 *
 * Verified here: identical answers to a brute-force scan, invariant P after every query,
 * multiset preservation, convergence to a sorted column under ascending single-value
 * probing, plus deterministic replay of a PRNG-generated workload.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

class CrackedColumn {
  constructor(values) {
    this.data = values.slice();
    this.original = values.slice();
    this.n = this.data.length;
    this.bounds = this.n > 0 ? [[-Infinity, Infinity]] : [];
    this.slots = this.n > 0 ? [[0, this.n]] : [];
    this.crackCount = 0;
    this.queryCount = 0;
  }

  get pieceCount() {
    return this.bounds.length;
  }

  /**
   * Answers `value in [low, high]` and cracks the pieces straddling an endpoint.
   * Returns the matching values in ascending order.
   */
  query(low, high) {
    this.queryCount++;
    const hits = [];
    if (this.n === 0 || low > high) return hits;
    const bounds = [];
    const slots = [];
    for (let p = 0; p < this.bounds.length; p++) {
      const lo = this.slots[p][0];
      const hi = this.slots[p][1];
      const pieceLow = this.bounds[p][0];
      const pieceHigh = this.bounds[p][1];
      if (pieceHigh < low || pieceLow > high) {
        bounds.push([pieceLow, pieceHigh]); // disjoint: touched no data
        slots.push([lo, hi]);
      } else if (low <= pieceLow && pieceHigh <= high) {
        bounds.push([pieceLow, pieceHigh]); // fully inside: all values match
        slots.push([lo, hi]);
        for (let i = lo; i < hi; i++) hits.push(this.data[i]);
      } else {
        // Boundary piece: crack into (< low), [low, high], (> high).
        const mid1 = this.partition(lo, hi, low, false);
        const mid2 = this.partition(mid1, hi, high, true);
        this.crackCount++;
        if (mid1 > lo) {
          bounds.push([pieceLow, Math.min(low - 1, pieceHigh)]);
          slots.push([lo, mid1]);
        }
        if (mid2 > mid1) {
          bounds.push([Math.max(low, pieceLow), Math.min(high, pieceHigh)]);
          slots.push([mid1, mid2]);
        }
        if (hi > mid2) {
          bounds.push([Math.max(high + 1, pieceLow), pieceHigh]);
          slots.push([mid2, hi]);
        }
        for (let i = mid1; i < mid2; i++) hits.push(this.data[i]);
      }
    }
    this.bounds = bounds;
    this.slots = slots;
    hits.sort((a, b) => a - b);
    return hits;
  }

  /**
   * In-place two-pointer partition of data[lo, hi) putting the values satisfying the
   * predicate first; returns the split index. `upperInclusive` selects `<= pivot` (true)
   * or `< pivot` (false). O(hi - lo) time, O(1) extra space, unstable.
   */
  partition(lo, hi, pivot, upperInclusive) {
    let i = lo;
    let j = hi;
    while (i < j) {
      const fits = upperInclusive ? this.data[i] <= pivot : this.data[i] < pivot;
      if (fits) {
        i++;
      } else {
        j--;
        const tmp = this.data[i];
        this.data[i] = this.data[j];
        this.data[j] = tmp;
      }
    }
    return i;
  }

  /** True when the pieces cover [0, n) contiguously and invariant P holds. */
  checkInvariants() {
    let covered = 0;
    for (let p = 0; p < this.bounds.length; p++) {
      const lo = this.slots[p][0];
      const hi = this.slots[p][1];
      if (lo !== covered || hi < lo) return false;
      covered = hi;
      for (let i = lo; i < hi; i++) {
        if (this.data[i] < this.bounds[p][0] || this.data[i] > this.bounds[p][1]) return false;
      }
      if (p + 1 < this.bounds.length && !(this.bounds[p][1] < this.bounds[p + 1][0])) return false;
    }
    if (covered !== this.n) return false;
    return this.multisetPreserved();
  }

  multisetPreserved() {
    const a = this.data.slice().sort((x, y) => x - y);
    const b = this.original.slice().sort((x, y) => x - y);
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }

  isSorted() {
    for (let i = 1; i < this.n; i++) {
      if (this.data[i - 1] > this.data[i]) return false;
    }
    return true;
  }
}

function bruteForceRange(values, low, high) {
  return values.filter((v) => v >= low && v <= high).sort((a, b) => a - b);
}

module.exports = { CrackedColumn, makeRng, bruteForceRange };

if (require.main === module) {
  // -- small hand-checked cases ------------------------------------------------------
  const empty = new CrackedColumn([]);
  deepStrictEqual(empty.query(0, 10), []);
  strictEqual(empty.pieceCount, 0);
  ok(empty.checkInvariants());

  const single = new CrackedColumn([42]);
  deepStrictEqual(single.query(42, 42), [42]);
  deepStrictEqual(single.query(43, 99), []);
  deepStrictEqual(single.query(0, 41), []);
  ok(single.checkInvariants());

  const col = new CrackedColumn([5, 3, 9, 3, 7, 5, 1, 9, 3, 7]);
  strictEqual(col.pieceCount, 1);
  deepStrictEqual(col.query(3, 7), [3, 3, 3, 5, 5, 7, 7]);
  ok(col.checkInvariants());
  strictEqual(col.crackCount, 1, "first query must crack the single initial piece");
  ok(col.pieceCount > 1, "cracking must create new pieces");
  deepStrictEqual(col.query(3, 7), [3, 3, 3, 5, 5, 7, 7], "repeat query keeps the answer");
  ok(col.checkInvariants());
  deepStrictEqual(col.query(1, 1), [1]);
  deepStrictEqual(col.query(9, 9), [9, 9]);
  deepStrictEqual(col.query(8, 8), [], "value never present");
  ok(col.checkInvariants());
  // low > high is a degenerate (empty) query and must not disturb the structure.
  deepStrictEqual(col.query(8, 2), []);
  ok(col.checkInvariants());

  // -- convergence: probing each distinct value ascending fully sorts the column ------
  const probe = new CrackedColumn([5, 3, 9, 3, 7, 5, 1, 9, 3, 7]);
  for (const v of [1, 3, 5, 7, 9]) probe.query(v, v);
  ok(probe.isSorted(), "ascending single-value probes must leave the column sorted");
  strictEqual(probe.pieceCount, 5, "one piece per distinct value");
  ok(probe.checkInvariants());

  // -- randomized cross-check against brute force -----------------------------------
  const rng = makeRng(20261);
  for (let trial = 0; trial < 12; trial++) {
    const n = 20 + rng(60);
    const values = Array.from({ length: n }, () => rng(21) - 10);
    const cr = new CrackedColumn(values);
    let previousPieces = cr.pieceCount;
    for (let q = 0; q < 25; q++) {
      const a = rng(31) - 15;
      const b = rng(31) - 15;
      const low = Math.min(a, b);
      const high = Math.max(a, b);
      deepStrictEqual(cr.query(low, high), bruteForceRange(values, low, high),
        `trial ${trial} query [${low},${high}]`);
      ok(cr.checkInvariants(), `invariant P broken after [${low},${high}]`);
      ok(cr.pieceCount >= previousPieces, "piece count must be monotone non-decreasing");
      previousPieces = cr.pieceCount;
    }
    ok(cr.pieceCount <= values.length, "never more pieces than rows");
  }

  // -- deterministic replay: same seed, same crack sequence -------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const values = Array.from({ length: 80 }, () => r(40));
    const cr = new CrackedColumn(values);
    const answers = [];
    for (let q = 0; q < 20; q++) {
      const lo = r(20);
      answers.push(cr.query(lo, lo + 10).join("|"));
    }
    return { answers, cracks: cr.crackCount, pieces: cr.pieceCount, data: cr.data.join(",") };
  }
  deepStrictEqual(replay(7), replay(7), "identical seeds must replay identically");
  ok(replay(7).cracks > 0, "the workload must actually crack the column");

  // -- duplicate-heavy column: cracking must not blow up the piece count -------------
  const duplicates = new CrackedColumn(Array.from({ length: 200 }, (_, i) => i % 4));
  for (let v = 0; v < 4; v++) deepStrictEqual(duplicates.query(v, v), new Array(50).fill(v));
  ok(duplicates.checkInvariants());
  strictEqual(duplicates.pieceCount, 4, "four distinct values, four pieces");

  console.log(`queries=${duplicates.queryCount + col.queryCount} cracks=${duplicates.crackCount + col.crackCount}`);
  console.log("PASS");
}
