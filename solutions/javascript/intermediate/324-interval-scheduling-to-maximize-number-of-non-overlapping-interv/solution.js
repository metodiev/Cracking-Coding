/** Task 324 - Interval scheduling to maximize number of non-overlapping intervals (greedy). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Greedy earliest finish time; touching endpoints count as overlapping (start > lastEnd). */
function maxNonOverlapping(intervals) {
  const ordered = intervals.map((iv) => iv.slice()).sort((u, v) => u[1] - v[1] || u[0] - v[0]);
  let count = 0;
  let lastEnd = null;
  for (const [start, end] of ordered) {
    if (lastEnd === null || start > lastEnd) {
      count++;
      lastEnd = end;
    }
  }
  return count;
}

/** Independent DP: best[i] = max(best[i-1], 1 + best[p(i)]) with p by binary search. */
function maxNonOverlappingDp(intervals) {
  if (intervals.length === 0) return 0;
  const ordered = intervals.map((iv) => iv.slice()).sort((u, v) => u[1] - v[1] || u[0] - v[0]);
  const finishes = ordered.map((iv) => iv[1]);
  const best = new Array(ordered.length + 1).fill(0);
  for (let i = 1; i <= ordered.length; i++) {
    const start = ordered[i - 1][0];
    let lo = 0;
    let hi = i - 1; // last j < i with finishes[j-1] < start
    while (lo < hi) {
      const mid = Math.floor((lo + hi + 1) / 2);
      if (finishes[mid - 1] < start) lo = mid;
      else hi = mid - 1;
    }
    const j = lo > 0 && finishes[lo - 1] < start ? lo : 0;
    best[i] = Math.max(best[i - 1], 1 + best[j]);
  }
  return best[ordered.length];
}

function lcgIntervals(count, lcg) {
  const intervals = [];
  for (let i = 0; i < count; i++) {
    const start = lcg.next(20);
    intervals.push([start, start + 1 + lcg.next(6)]);
  }
  return intervals;
}

module.exports = { maxNonOverlapping, maxNonOverlappingDp, lcgIntervals, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // classic example: [1,2] and [2,3] touch, so they conflict; best is 2
  assert.strictEqual(maxNonOverlapping([[1, 2], [2, 3], [3, 4], [1, 3]]), 2);
  assert.strictEqual(maxNonOverlappingDp([[1, 2], [2, 3], [3, 4], [1, 3]]), 2);

  // chain with gaps: all four fit; the wide decoy is not worth taking
  const chain = [[1, 2], [3, 4], [5, 6], [7, 8], [1, 8]];
  assert.strictEqual(maxNonOverlapping(chain), 4);
  assert.strictEqual(maxNonOverlappingDp(chain), 4);
  assert.strictEqual(maxNonOverlapping([[1, 2], [3, 4], [5, 6], [7, 8]]), 4);

  assert.strictEqual(maxNonOverlapping([]), 0);
  assert.strictEqual(maxNonOverlapping([[1, 5]]), 1);
  assert.strictEqual(maxNonOverlapping([[1, 3], [2, 4], [3, 5], [4, 6]]), 2);
  assert.strictEqual(maxNonOverlapping([[1, 2], [1, 2], [1, 2]]), 1);
  assert.strictEqual(maxNonOverlapping([[1, 4], [4, 4]]), 1); // touching conflicts

  const lcg = new Lcg(909);
  for (let trial = 0; trial < 400; trial++) {
    const count = 1 + lcg.next(9);
    const intervals = lcgIntervals(count, lcg);
    assert.strictEqual(
      maxNonOverlapping(intervals),
      maxNonOverlappingDp(intervals),
      `trial ${trial}: ${JSON.stringify(intervals)}`
    );
  }
  console.log("PASS");
}
